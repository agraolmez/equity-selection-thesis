"""Independent verification and immutable authority publication for Stage 8.

The verifier has two deliberately separate layers:

* a byte-for-byte replay of the complete primary materialization; and
* independently written scientific checks for the risk and conservative-cost
  formulae.  The latter do not call the numerical helpers in ``risk_math`` or
  ``cost_math``.

Only a zero-difference result may replace ``LATEST_RISK_COST_VERIFIED.json``.
That pointer authorizes Stage 9 model training and nothing later.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import numbers
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from portfolio_bot.data.secure_publication import (
    SecurePublicationError,
    SecureStagingDirectory,
    replace_pointer_last,
)
from portfolio_bot.risk_cost.contract import (
    DEFAULT_POLICY,
    FACTOR_NAMES,
    PROJECT_ROOT,
    canonical_json_bytes,
    project_path,
    published_schemas,
    schema_sha256,
    sha256_file,
    validate_contract,
)
from portfolio_bot.risk_cost.inputs import Stage8Inputs, resolve_stage8_inputs
from portfolio_bot.risk_cost.materialization import materialize_risk_cost
from portfolio_bot.risk_cost.publication import compute_build_identity


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_BUILD_ID = re.compile(r"phase04-risk-cost-[0-9a-f]{24}\Z")
_VERIFY_ID = re.compile(r"verify-[0-9a-f]{16}\Z")
_PRIMARY_STATUS = "PRIMARY_PASSED_PENDING_INDEPENDENT_VERIFICATION"
_VERIFIED_STATUS = "VERIFIED_PASSED"
_MISSING_DAY = np.iinfo(np.int32).min
_PRIMARY_AUTHORIZATIONS = {
    "independent_verification_completed": False,
    "models_portfolios_pnl_or_backtests_authorized": False,
    "stage_09_model_training_authorized": False,
    "stage_10_or_later_authorized": False,
    "validation_or_final_test_access_authorized": False,
}
_VERIFIED_AUTHORIZATIONS = {
    "independent_verification_completed": True,
    "models_portfolios_pnl_or_backtests_authorized": False,
    "stage_09_model_training_authorized": True,
    "stage_10_or_later_authorized": False,
    "validation_or_final_test_access_authorized": False,
}
_VERIFICATION_CODE_PATHS = (
    "scripts/verify_stage_08_risk_cost.py",
    "src/portfolio_bot/risk_cost/independent_verification.py",
)


class Stage8VerificationError(RuntimeError):
    """Raised when Stage 8 cannot be independently verified safely."""


@dataclass(frozen=True)
class Difference:
    code: str
    expected: Any
    observed: Any
    details: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "expected": _json_safe(self.expected),
            "observed": _json_safe(self.observed),
            "details": self.details,
        }


@dataclass(frozen=True)
class AuthenticatedPrimary:
    policy_path: Path
    policy: Mapping[str, Any]
    rules: Mapping[str, Any]
    inputs: Stage8Inputs
    primary_pointer_path: Path
    primary_pointer_sha256: str
    primary_pointer: Mapping[str, Any]
    primary_manifest_path: Path
    primary_manifest_sha256: str
    manifest: Mapping[str, Any]
    output_paths: Mapping[str, Mapping[str, Path]]
    output_hashes: Mapping[str, Mapping[str, str]]
    stat_identities: Mapping[str, tuple[int, int, int, int]] = field(repr=False)

    @property
    def build_id(self) -> str:
        return str(self.manifest["build_id"])


@dataclass(frozen=True)
class VerificationResult:
    passed: bool
    build_id: str
    verification_id: str | None
    checks: Mapping[str, Any]
    differences: tuple[Difference, ...]
    verification_path: Path | None
    verified_pointer_path: Path | None


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Stage8VerificationError(message)


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (dt.date, dt.datetime)):
        return value.isoformat()
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        result = float(value)
        _require(math.isfinite(result), "Non-finite verification value")
        return result
    raise Stage8VerificationError(f"Unsupported verification value: {type(value)!r}")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_canonical_json(path: Path, label: str) -> dict[str, Any]:
    _require(path.is_file() and not path.is_symlink(), f"Unsafe {label} path")
    payload = path.read_bytes()
    try:
        value = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise Stage8VerificationError(f"Invalid {label} JSON") from exc
    _require(isinstance(value, dict), f"{label} is not an object")
    _require(payload == canonical_json_bytes(value), f"{label} is not canonical JSON")
    return value


def _stat_identity(path: Path) -> tuple[int, int, int, int]:
    value = path.stat(follow_symlinks=False)
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)


def _safe_member(root: Path, path: Path, label: str) -> Path:
    root_resolved = root.resolve(strict=True)
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise Stage8VerificationError(f"{label} escaped its trusted root") from exc
    cursor = resolved
    while cursor != root_resolved:
        _require(not cursor.is_symlink(), f"Symlinked {label} path component")
        cursor = cursor.parent
    _require(resolved.is_file() and not resolved.is_symlink(), f"Unsafe {label}")
    return resolved


def _file_receipt(path: Path, *, parquet: bool) -> dict[str, Any]:
    before = _stat_identity(path)
    digest = sha256_file(path)
    after = _stat_identity(path)
    _require(before == after, f"File changed while hashing: {path}")
    if parquet:
        metadata = pq.ParquetFile(path).metadata
        schema = pq.ParquetFile(path).schema_arrow
        rows: int | None = int(metadata.num_rows)
        groups: int | None = int(metadata.num_row_groups)
        schema_hash: str | None = schema_sha256(schema)
    else:
        rows = groups = schema_hash = None
    return {
        "path": path.name,
        "sha256": digest,
        "size_bytes": path.stat().st_size,
        "rows": rows,
        "row_groups": groups,
        "schema_sha256": schema_hash,
    }


def _authenticate_outputs(
    *,
    policy: Mapping[str, Any],
    manifest: Mapping[str, Any],
    database_root: Path,
) -> tuple[
    dict[str, dict[str, Path]],
    dict[str, dict[str, str]],
    dict[str, tuple[int, int, int, int]],
]:
    schemas = published_schemas()
    paths: dict[str, dict[str, Path]] = {}
    hashes: dict[str, dict[str, str]] = {}
    identities: dict[str, tuple[int, int, int, int]] = {}
    for location in ("interim", "processed", "quality"):
        specification = manifest["outputs"][location]
        expected_relative = (
            Path(str(policy["database"][f"{location}_build_root"]))
            / str(manifest["build_id"])
        )
        _require(
            Path(str(specification["path"])) == expected_relative,
            f"{location} output path differs",
        )
        directory = (database_root / expected_relative).resolve(strict=True)
        _require(
            directory.is_dir() and not directory.is_symlink(),
            f"Unsafe {location} output directory",
        )
        required = tuple(policy["outputs"][location]["required_files"])
        observed = {item.name for item in directory.iterdir()}
        _require(observed == set(required), f"{location} closed output set differs")
        expected_payloads = set(required) - {"_SUCCESS.json"}
        declared = specification["files"]
        _require(
            isinstance(declared, Mapping) and set(declared) == expected_payloads,
            f"{location} manifest payload set differs",
        )
        location_paths: dict[str, Path] = {}
        location_hashes: dict[str, str] = {}
        for name in required:
            member = _safe_member(database_root, directory / name, f"{location}/{name}")
            receipt = _file_receipt(member, parquet=name.endswith(".parquet"))
            if name == "_SUCCESS.json":
                _require(
                    receipt["sha256"] == specification["success_sha256"],
                    f"{location} success hash differs",
                )
            else:
                _require(dict(declared[name]) == receipt, f"{location}/{name} receipt differs")
                if name.endswith(".parquet"):
                    expected_schema = schemas[name]
                    actual_schema = pq.ParquetFile(member).schema_arrow
                    _require(
                        actual_schema.equals(expected_schema, check_metadata=True),
                        f"{location}/{name} schema differs",
                    )
                else:
                    _read_canonical_json(member, f"{location}/{name}")
            location_paths[name] = member
            location_hashes[name] = str(receipt["sha256"])
            identities[f"{location}/{name}"] = _stat_identity(member)

        expected_success = {
            "schema_version": 1,
            "contract_id": policy["contract_id"],
            "policy_id": policy["policy_id"],
            "build_id": manifest["build_id"],
            "content_fingerprint": manifest["bindings"]["content_fingerprint"],
            "location": location,
            "status": f"PRIMARY_{location.upper()}_COMPLETE",
            "files": {name: dict(declared[name]) for name in sorted(declared)},
        }
        observed_success = _read_canonical_json(
            location_paths["_SUCCESS.json"], f"{location} success receipt"
        )
        _require(observed_success == expected_success, f"{location} success receipt differs")
        paths[location] = location_paths
        hashes[location] = location_hashes
    return paths, hashes, identities


def authenticate_primary(
    *,
    policy_path: Path = DEFAULT_POLICY,
    primary_pointer_path: Path | None = None,
) -> AuthenticatedPrimary:
    """Authenticate the complete primary graph before any recomputation."""

    policy_path = policy_path.expanduser().resolve()
    policy, rules = validate_contract(policy_path)
    inputs = resolve_stage8_inputs(policy_path)
    pointer_path = (
        project_path(policy["publication"]["latest_primary_pointer"])
        if primary_pointer_path is None
        else primary_pointer_path.expanduser().resolve()
    )
    pointer = _read_canonical_json(pointer_path, "Stage 8 primary pointer")
    _require(
        set(pointer)
        == {
            "schema_version",
            "contract_id",
            "policy_id",
            "build_id",
            "status",
            "primary_manifest_sha256",
            "authorizations",
        },
        "Primary pointer key set differs",
    )
    _require(
        pointer["schema_version"] == 1
        and pointer["contract_id"] == policy["contract_id"]
        and pointer["policy_id"] == policy["policy_id"]
        and pointer["status"] == _PRIMARY_STATUS
        and pointer["authorizations"] == _PRIMARY_AUTHORIZATIONS
        and _BUILD_ID.fullmatch(str(pointer["build_id"])) is not None
        and _SHA256.fullmatch(str(pointer["primary_manifest_sha256"])) is not None,
        "Primary pointer identity or authority differs",
    )
    artifact_root = project_path(policy["publication"]["artifact_root"])
    manifest_path = (
        artifact_root
        / str(policy["publication"]["primary_build_namespace"])
        / str(pointer["build_id"])
        / "manifest.json"
    )
    manifest_path = _safe_member(PROJECT_ROOT, manifest_path, "Stage 8 primary manifest")
    manifest_hash = sha256_file(manifest_path)
    _require(
        manifest_hash == pointer["primary_manifest_sha256"],
        "Primary manifest hash differs",
    )
    manifest = _read_canonical_json(manifest_path, "Stage 8 primary manifest")
    _require(
        set(manifest)
        == {
            "schema_version",
            "contract_id",
            "policy_id",
            "build_id",
            "status",
            "authorizations",
            "bindings",
            "inputs",
            "outputs",
            "summary",
            "hard_gates",
        },
        "Primary manifest key set differs",
    )
    _require(
        manifest["schema_version"] == 1
        and manifest["contract_id"] == policy["contract_id"]
        and manifest["policy_id"] == policy["policy_id"]
        and manifest["build_id"] == pointer["build_id"]
        and manifest["status"] == _PRIMARY_STATUS,
        "Primary manifest identity differs",
    )
    _require(
        manifest["authorizations"]
        == {
            "independent_verification_required": True,
            "models_portfolios_pnl_or_backtests_authorized": False,
            "stage_09_model_training_authorized": False,
            "stage_10_or_later_authorized": False,
            "validation_or_final_test_access_authorized": False,
        },
        "Primary manifest authority differs",
    )
    _require(
        manifest["inputs"]["table_receipts"] == list(inputs.input_receipts)
        and manifest["inputs"]["authority_build_id"] == inputs.authority["build_id"]
        and manifest["inputs"]["authority_verification_id"]
        == inputs.authority["verification_id"]
        and manifest["inputs"]["label_payload_rows_read"] == 0,
        "Primary input bindings differ",
    )
    identity = compute_build_identity(inputs, policy_path=policy_path)
    _require(
        identity.build_id == manifest["build_id"]
        and identity.content_fingerprint == manifest["bindings"]["content_fingerprint"]
        and dict(identity.code_hashes) == manifest["bindings"]["code_hashes"]
        and dict(identity.configuration_hashes)
        == manifest["bindings"]["configuration_hashes"]
        and identity.input_receipts_sha256
        == manifest["bindings"]["input_receipts_sha256"],
        "Primary deterministic build identity differs",
    )
    _require(
        manifest["hard_gates"] == policy["quality_gates"],
        "Primary hard gates differ from frozen policy",
    )
    output_paths, output_hashes, stat_identities = _authenticate_outputs(
        policy=policy,
        manifest=manifest,
        database_root=inputs.database_root,
    )
    stat_identities["primary_pointer"] = _stat_identity(pointer_path)
    stat_identities["primary_manifest"] = _stat_identity(manifest_path)
    return AuthenticatedPrimary(
        policy_path=policy_path,
        policy=policy,
        rules=rules,
        inputs=inputs,
        primary_pointer_path=pointer_path,
        primary_pointer_sha256=sha256_file(pointer_path),
        primary_pointer=pointer,
        primary_manifest_path=manifest_path,
        primary_manifest_sha256=manifest_hash,
        manifest=manifest,
        output_paths=output_paths,
        output_hashes=output_hashes,
        stat_identities=stat_identities,
    )


def _date_days(array: pa.Array | pa.ChunkedArray) -> np.ndarray:
    cast = pc.cast(array, pa.int32())
    filled = pc.fill_null(cast, pa.scalar(_MISSING_DAY, pa.int32()))
    return filled.to_numpy(zero_copy_only=False).astype(np.int32, copy=False)


def _floats(array: pa.Array | pa.ChunkedArray) -> np.ndarray:
    return pc.cast(array, pa.float64()).to_numpy(zero_copy_only=False).astype(
        np.float64, copy=False
    )


def _nullable_equal(
    observed: np.ndarray,
    expected: np.ndarray,
    *,
    atol: float = 1e-12,
    rtol: float = 1e-10,
) -> np.ndarray:
    return np.isclose(observed, expected, atol=atol, rtol=rtol, equal_nan=True)


def independent_factor_covariance(
    factor_matrix: np.ndarray,
    ages: np.ndarray,
    *,
    half_life: float = 63.0,
) -> np.ndarray | None:
    """Independent EW population covariance (no primary numerical helper)."""

    matrix = np.asarray(factor_matrix, dtype=np.float64)
    age = np.asarray(ages, dtype=np.float64)
    if (
        matrix.ndim != 2
        or matrix.shape[1] != 6
        or matrix.shape[0] < 126
        or age.shape != (matrix.shape[0],)
        or not np.isfinite(matrix).all()
        or not np.isfinite(age).all()
    ):
        return None
    weights = np.power(2.0, -age / half_life)
    weight_sum = np.sum(weights, dtype=np.float64)
    mean = np.sum(matrix * weights[:, None], axis=0) / weight_sum
    centered = matrix - mean
    covariance = centered.T @ (centered * weights[:, None]) / weight_sum
    return (covariance + covariance.T) * 0.5


def independent_ewls(
    security_returns: np.ndarray,
    risk_free: np.ndarray,
    factor_matrix: np.ndarray,
    ages: np.ndarray,
) -> tuple[np.ndarray, float] | None:
    """Independent frozen EWLS coefficients and population residual variance."""

    returns = np.asarray(security_returns, dtype=np.float64)
    rf = np.asarray(risk_free, dtype=np.float64)
    factors = np.asarray(factor_matrix, dtype=np.float64)
    age = np.asarray(ages, dtype=np.float64)
    if (
        returns.ndim != 1
        or returns.size < 126
        or rf.shape != returns.shape
        or factors.shape != (returns.size, 6)
        or age.shape != returns.shape
        or not np.isfinite(returns).all()
        or not np.isfinite(rf).all()
        or not np.isfinite(factors).all()
    ):
        return None
    weights = np.power(2.0, -age / 63.0)
    roots = np.sqrt(weights)
    design = np.column_stack((np.ones(returns.size, dtype=np.float64), factors))
    coefficients, _, rank, _ = np.linalg.lstsq(
        design * roots[:, None], (returns - rf) * roots, rcond=1e-12
    )
    if rank != 7:
        return None
    residual = returns - rf - design @ coefficients
    variance = np.sum(weights * residual * residual, dtype=np.float64) / np.sum(
        weights, dtype=np.float64
    )
    return coefficients, float(variance)


def independent_impact_volatility(
    latest_20: Sequence[float | None], latest_60: Sequence[float | None]
) -> tuple[float, str, int] | None:
    """Independent 20-session/60-session daily standard-deviation fallback."""

    for window, minimum, values, source in (
        (20, 15, latest_20, "20_SESSION"),
        (60, 40, latest_60, "60_SESSION"),
    ):
        sequence = list(values)
        if len(sequence) < window:
            continue
        vector = np.asarray(
            [float(item) for item in sequence[-window:] if item is not None and math.isfinite(float(item))],
            dtype=np.float64,
        )
        if vector.size >= minimum:
            return float(np.std(vector, ddof=1)), source, int(vector.size)
    return None


def independent_liquidity_quintiles(
    permnos: Sequence[int], adv: Sequence[float | None]
) -> np.ndarray:
    """Independent deterministic lowest-to-highest ADV quintiles."""

    _require(len(permnos) == len(adv), "Quintile inputs differ")
    size = len(permnos)

    def key(index: int) -> tuple[int, float, int]:
        value = adv[index]
        if value is None or not math.isfinite(float(value)) or float(value) <= 0.0:
            return (0, 0.0, int(permnos[index]))
        return (1, float(value), int(permnos[index]))

    output = np.empty(size, dtype=np.int8)
    for rank, index in enumerate(sorted(range(size), key=key)):
        output[index] = math.floor(5 * rank / size) + 1
    return output


def _factor_context(inputs: Stage8Inputs) -> dict[str, Any]:
    sessions = inputs.read_trading_calendar()
    session_days = _date_days(sessions["calendar_date"])
    factors = inputs.read_factors()
    factor_days = _date_days(factors["observation_date"])
    positions = np.searchsorted(session_days, factor_days)
    _require(
        np.all(positions < session_days.size)
        and np.array_equal(session_days[positions], factor_days),
        "Factor observations are not exact sessions",
    )
    matrix = np.full((session_days.size, 6), np.nan, dtype=np.float64)
    rf = np.full(session_days.size, np.nan, dtype=np.float64)
    availability = np.full(session_days.size, _MISSING_DAY, dtype=np.int32)
    for ordinal, name in enumerate(FACTOR_NAMES):
        matrix[positions, ordinal] = _floats(factors[name])
    rf[positions] = _floats(factors["rf"])
    availability[positions] = _date_days(factors["public_availability_date"])
    return {
        "session_days": session_days,
        "matrix": matrix,
        "rf": rf,
        "availability": availability,
    }


def _factor_scientific_check(
    artifacts: AuthenticatedPrimary,
    context: Mapping[str, Any],
) -> tuple[dict[str, Any], list[Difference], dict[int, np.ndarray]]:
    table = pq.read_table(
        artifacts.output_paths["processed"]["development_factor_covariance.parquet"],
        use_threads=False,
    )
    signal_days = _date_days(table["signal_date"])
    unique = np.unique(signal_days)
    sessions = context["session_days"]
    matrix = context["matrix"]
    rf = context["rf"]
    availability = context["availability"]
    mismatches = 0
    covariance_by_day: dict[int, np.ndarray] = {}
    for signal in unique:
        day = int(signal)
        signal_position = int(np.searchsorted(sessions, day))
        start = max(0, signal_position - 251)
        indices = np.arange(start, signal_position + 1, dtype=np.int32)
        complete = (
            np.isfinite(matrix[indices]).all(axis=1)
            & np.isfinite(rf[indices])
            & (availability[indices] != _MISSING_DAY)
            & (availability[indices] <= day)
        )
        selected = indices[complete]
        covariance = independent_factor_covariance(
            matrix[selected], signal_position - selected
        )
        rows = np.flatnonzero(signal_days == day)
        _require(rows.size == 36, f"Factor covariance domain differs for {day}")
        observed_count = table["factor_observation_count"].take(pa.array(rows)).to_numpy()
        if not np.all(observed_count == selected.size):
            mismatches += int(np.sum(observed_count != selected.size))
        first_expected = None if selected.size == 0 else int(sessions[selected[0]])
        last_expected = None if selected.size == 0 else int(sessions[selected[-1]])
        first_observed = _date_days(table["factor_window_first_date"].take(pa.array(rows)))
        last_observed = _date_days(table["factor_window_last_date"].take(pa.array(rows)))
        first_value = _MISSING_DAY if first_expected is None else first_expected
        last_value = _MISSING_DAY if last_expected is None else last_expected
        mismatches += int(np.sum(first_observed != first_value))
        mismatches += int(np.sum(last_observed != last_value))
        observed_daily = _floats(table["covariance_daily"].take(pa.array(rows)))
        observed_monthly = _floats(table["covariance_monthly"].take(pa.array(rows)))
        if covariance is None:
            mismatches += int(np.sum(np.isfinite(observed_daily)))
        else:
            covariance_by_day[day] = covariance
            expected = covariance.reshape(-1)
            mismatches += int(np.sum(~_nullable_equal(observed_daily, expected)))
            mismatches += int(np.sum(~_nullable_equal(observed_monthly, 21.0 * expected)))
            row_ord = table["factor_row_ordinal"].take(pa.array(rows)).to_numpy()
            col_ord = table["factor_col_ordinal"].take(pa.array(rows)).to_numpy()
            mismatches += int(np.sum(row_ord != np.repeat(np.arange(6), 6)))
            mismatches += int(np.sum(col_ord != np.tile(np.arange(6), 6)))
    differences: list[Difference] = []
    if mismatches:
        differences.append(
            Difference(
                "factor_covariance_independent_recomputation",
                0,
                mismatches,
                "All 178 six-factor covariance matrices, windows, and monthly scales were independently recomputed.",
            )
        )
    return {
        "factor_signal_dates_checked": int(unique.size),
        "factor_covariance_cells_checked": int(table.num_rows),
        "factor_covariance_independent_mismatches": mismatches,
    }, differences, covariance_by_day


def _stratified_pairs(processed: pa.Table) -> list[tuple[int, int, int]]:
    days = _date_days(processed["signal_date"])
    permnos = processed["permno"].to_numpy(zero_copy_only=False).astype(np.int64)
    pairs: list[tuple[int, int, int]] = []
    for day in np.unique(days):
        indices = np.flatnonzero(days == day)
        for ordinal in sorted({0, len(indices) // 2, len(indices) - 1}):
            row = int(indices[ordinal])
            pairs.append((int(day), int(permnos[row]), row))
    return pairs


def _collect_selected_histories(
    inputs: Stage8Inputs,
    selected_permnos: Sequence[int],
    *,
    batch_rows: int,
    progress: Callable[[str], None],
) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    selected = np.asarray(sorted(set(selected_permnos)), dtype=np.int64)
    pieces: dict[int, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {
        int(value): [] for value in selected
    }
    last_signal = int((dt.date(2014, 10, 31) - dt.date(1970, 1, 1)).days)
    delivered = 0
    for batch_number, batch in enumerate(
        inputs.iter_return_batches(batch_rows=batch_rows), start=1
    ):
        delivered += batch.num_rows
        names = batch.schema.names
        column = lambda name: batch.column(names.index(name))
        permnos = column("permno").to_numpy(zero_copy_only=False).astype(np.int64)
        days = _date_days(column("calendar_date"))
        span_end = _date_days(column("return_span_end_date"))
        availability = _date_days(column("outcome_availability_date"))
        covered = column("covered_trading_sessions").to_numpy(zero_copy_only=False)
        usable = pc.fill_null(column("return_history_usable"), False).to_numpy(
            zero_copy_only=False
        )
        returns = _floats(column("authoritative_total_return"))
        mask = (
            usable
            & (covered == 1)
            & (span_end == days)
            & (days <= last_signal)
            & (availability != _MISSING_DAY)
            & (availability <= last_signal)
            & np.isfinite(returns)
            & (returns >= -1.0)
            & np.isin(permnos, selected, assume_unique=False)
        )
        if np.any(mask):
            p = permnos[mask]
            d = days[mask]
            a = availability[mask]
            r = returns[mask]
            for permno in np.unique(p):
                own = p == permno
                pieces[int(permno)].append((d[own], a[own], r[own]))
        if batch_number % 25 == 0:
            progress(
                f"independent sample history scan batches={batch_number}; rows={delivered:,}"
            )
    result: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for permno, groups in pieces.items():
        if not groups:
            result[permno] = (
                np.asarray([], dtype=np.int32),
                np.asarray([], dtype=np.int32),
                np.asarray([], dtype=np.float64),
            )
            continue
        days = np.concatenate([item[0] for item in groups])
        availability = np.concatenate([item[1] for item in groups])
        returns = np.concatenate([item[2] for item in groups])
        order = np.lexsort((days,))
        result[permno] = (days[order], availability[order], returns[order])
    return result


def _global_window(
    session_positions: np.ndarray,
    availability: np.ndarray,
    returns: np.ndarray,
    *,
    signal_position: int,
    signal_day: int,
    sessions: int,
) -> list[float | None]:
    if signal_position + 1 < sessions:
        return []
    start = signal_position - sessions + 1
    output: list[float | None] = [None] * sessions
    left = np.searchsorted(session_positions, start, side="left")
    right = np.searchsorted(session_positions, signal_position, side="right")
    for position, available, value in zip(
        session_positions[left:right],
        availability[left:right],
        returns[left:right],
        strict=True,
    ):
        if int(available) <= signal_day and math.isfinite(float(value)):
            output[int(position) - start] = float(value)
    return output


def _stratified_security_check(
    artifacts: AuthenticatedPrimary,
    processed: pa.Table,
    audit: pa.Table,
    context: Mapping[str, Any],
    *,
    batch_rows: int,
    progress: Callable[[str], None],
) -> tuple[dict[str, Any], list[Difference]]:
    pairs = _stratified_pairs(processed)
    histories = _collect_selected_histories(
        artifacts.inputs,
        [permno for _, permno, _ in pairs],
        batch_rows=batch_rows,
        progress=progress,
    )
    sessions = context["session_days"]
    factors = context["matrix"]
    rf = context["rf"]
    factor_availability = context["availability"]
    coefficient_names = ["beta_risk_intercept"] + [
        f"beta_risk_{name}" for name in FACTOR_NAMES
    ]
    observed_coefficients = np.column_stack(
        [_floats(processed[name]) for name in coefficient_names]
    )
    observed_raw = _floats(processed["specific_variance_raw_daily"])
    observed_impact = _floats(processed["impact_volatility_daily"])
    observed_impact_source = np.asarray(
        processed["impact_volatility_source"].to_pylist(), dtype=object
    )
    observed_impact_n = processed["impact_volatility_observations"].to_numpy(
        zero_copy_only=False
    )
    observed_risk_n = processed["risk_observation_count"].to_numpy(
        zero_copy_only=False
    )
    coefficient_mismatches = 0
    residual_mismatches = 0
    observation_mismatches = 0
    impact_mismatches = 0
    audited_dates: set[int] = set()
    for signal_day, permno, row in pairs:
        audited_dates.add(signal_day)
        days, availability, returns = histories[permno]
        positions = np.searchsorted(sessions, days)
        exact = (positions < sessions.size) & (
            sessions[np.minimum(positions, sessions.size - 1)] == days
        )
        positions = positions[exact].astype(np.int32, copy=False)
        availability = availability[exact]
        returns = returns[exact]
        signal_position = int(np.searchsorted(sessions, signal_day))
        start = max(0, signal_position - 251)
        left = int(np.searchsorted(positions, start, side="left"))
        right = int(np.searchsorted(positions, signal_position, side="right"))
        selected_positions = positions[left:right]
        selected_availability = availability[left:right]
        selected_returns = returns[left:right]
        complete = (
            (selected_availability <= signal_day)
            & np.isfinite(selected_returns)
            & np.isfinite(rf[selected_positions])
            & np.isfinite(factors[selected_positions]).all(axis=1)
            & (factor_availability[selected_positions] != _MISSING_DAY)
            & (factor_availability[selected_positions] <= signal_day)
        )
        selected_positions = selected_positions[complete]
        selected_returns = selected_returns[complete]
        observation_mismatches += int(
            int(observed_risk_n[row]) != int(selected_positions.size)
        )
        fit = independent_ewls(
            selected_returns,
            rf[selected_positions],
            factors[selected_positions],
            signal_position - selected_positions,
        )
        if fit is None:
            coefficient_mismatches += int(np.isfinite(observed_coefficients[row]).sum())
            residual_mismatches += int(math.isfinite(float(observed_raw[row])))
        else:
            coefficients, variance = fit
            coefficient_mismatches += int(
                np.sum(~_nullable_equal(observed_coefficients[row], coefficients))
            )
            residual_mismatches += int(
                not math.isclose(
                    float(observed_raw[row]), variance, rel_tol=1e-10, abs_tol=1e-12
                )
            )

        window20 = _global_window(
            positions,
            availability,
            returns,
            signal_position=signal_position,
            signal_day=signal_day,
            sessions=20,
        )
        window60 = _global_window(
            positions,
            availability,
            returns,
            signal_position=signal_position,
            signal_day=signal_day,
            sessions=60,
        )
        impact = independent_impact_volatility(window20, window60)
        if impact is None:
            impact_mismatches += int(math.isfinite(float(observed_impact[row])))
        else:
            value, source, count = impact
            impact_mismatches += int(
                not math.isclose(
                    float(observed_impact[row]), value, rel_tol=1e-10, abs_tol=1e-12
                )
            )
            impact_mismatches += int(observed_impact_source[row] != source)
            impact_mismatches += int(int(observed_impact_n[row]) != count)
    differences: list[Difference] = []
    total_mismatches = (
        coefficient_mismatches
        + residual_mismatches
        + observation_mismatches
        + impact_mismatches
    )
    # Five early development dates have no long-eligible scoring security.  They
    # cannot supply an EWLS pair, so independently close those dates by proving
    # that every audit row is fail-closed and contains no risk/cost estimate.
    audit_days = _date_days(audit["signal_date"])
    all_signal_dates = {int(value) for value in np.unique(audit_days)}
    empty_dates = sorted(all_signal_dates - audited_dates)
    empty_date_mismatches = 0
    audit_status = np.asarray(audit["risk_status"].to_pylist(), dtype=object)
    audit_long = audit["long_eligible"].to_numpy(zero_copy_only=False)
    audit_new_long = audit["new_long_trade_allowed"].to_numpy(zero_copy_only=False)
    audit_beta = _floats(audit["beta_risk_intercept"])
    audit_raw = _floats(audit["specific_variance_raw_daily"])
    for day in empty_dates:
        idx = np.flatnonzero(audit_days == day)
        empty_date_mismatches += int(
            np.sum(audit_long[idx])
            + np.sum(audit_new_long[idx])
            + np.sum(audit_status[idx] != "NOT_LONG_ELIGIBLE")
            + np.sum(np.isfinite(audit_beta[idx]))
            + np.sum(np.isfinite(audit_raw[idx]))
        )
    total_mismatches += empty_date_mismatches
    if total_mismatches:
        differences.append(
            Difference(
                "stratified_security_signal_recomputation",
                0,
                total_mismatches,
                "Three deterministic PERMNO strata were independently checked on every signal date.",
            )
        )
    return {
        "security_signal_pairs_checked": len(pairs),
        "security_signal_dates_with_eligible_strata": len(audited_dates),
        "security_signal_dates_without_eligible_rows": len(empty_dates),
        "security_signal_date_domain_checked": len(audited_dates | set(empty_dates)),
        "empty_signal_date_fail_closed_mismatches": empty_date_mismatches,
        "ewls_coefficient_mismatches": coefficient_mismatches,
        "raw_residual_variance_mismatches": residual_mismatches,
        "risk_observation_count_mismatches": observation_mismatches,
        "impact_volatility_sample_mismatches": impact_mismatches,
    }, differences


def _full_vector_checks(
    processed: pa.Table,
    covariance_by_day: Mapping[int, np.ndarray],
) -> tuple[dict[str, Any], list[Difference]]:
    days = _date_days(processed["signal_date"])
    permnos = processed["permno"].to_numpy(zero_copy_only=False).astype(np.int64)
    raw = _floats(processed["specific_variance_raw_daily"])
    nobs = processed["risk_observation_count"].to_numpy(zero_copy_only=False)
    sectors = np.asarray(processed["ff12_sector_category"].to_pylist(), dtype=object)
    target_observed = _floats(processed["shrinkage_target_daily"])
    source_observed = np.asarray(
        processed["shrinkage_target_source"].to_pylist(), dtype=object
    )
    population_observed = processed["shrinkage_population_count"].to_numpy(
        zero_copy_only=False
    )
    shrunk_observed = _floats(processed["specific_variance_shrunk_daily"])
    ridge_observed = _floats(processed["risk_ridge_daily"])
    specific_daily = _floats(processed["specific_variance_daily"])
    specific_monthly = _floats(processed["specific_variance_monthly"])
    security_pre = _floats(processed["security_variance_pre_ridge_daily"])
    security_daily = _floats(processed["security_variance_daily"])
    security_monthly = _floats(processed["security_variance_monthly"])
    betas = np.column_stack([_floats(processed[f"beta_risk_{name}"]) for name in FACTOR_NAMES])
    short = pc.fill_null(processed["short_eligible"], False).to_numpy(
        zero_copy_only=False
    )
    spread = _floats(processed["full_spread_20d"])
    adv = _floats(processed["adv20_usd"])
    impact = _floats(processed["impact_volatility_daily"])
    exit_vol = _floats(processed["exit_impact_volatility_daily"])
    exit_source = np.asarray(processed["exit_volatility_source"].to_pylist(), dtype=object)
    cost_status = np.asarray(processed["cost_input_status"].to_pylist(), dtype=object)
    new_long = processed["new_long_trade_allowed"].to_numpy(zero_copy_only=False)
    new_short = processed["new_short_trade_allowed"].to_numpy(zero_copy_only=False)
    quintile_py = processed["short_liquidity_quintile"].to_pylist()
    quintile = np.asarray([0 if value is None else int(value) for value in quintile_py], dtype=np.int8)
    primary_borrow = _floats(processed["primary_borrow_rate_annual"])
    stress_borrow = _floats(processed["stress_borrow_rate_annual"])

    shrinkage_mismatches = ridge_mismatches = monthly_mismatches = 0
    permission_mismatches = borrow_mismatches = exit_mismatches = 0
    for day in np.unique(days):
        idx = np.flatnonzero(days == day)
        valid = np.isfinite(raw[idx]) & (raw[idx] >= 0.0) & (nobs[idx] >= 126)
        values = raw[idx][valid]
        _require(values.size > 0, f"Missing shrinkage population for {day}")
        cross_median = float(np.quantile(values, 0.5, method="linear"))
        expected_target = np.full(idx.size, cross_median, dtype=np.float64)
        expected_source = np.full(idx.size, "CROSS_SECTION", dtype=object)
        expected_population = np.full(idx.size, values.size, dtype=np.int32)
        normalized = np.asarray(
            [
                None
                if value is None or not str(value).strip() or str(value).strip().upper() == "UNKNOWN"
                else str(value).strip()
                for value in sectors[idx]
            ],
            dtype=object,
        )
        for sector in sorted({value for value in normalized if value is not None}):
            sector_mask = normalized == sector
            sector_values = raw[idx][sector_mask & valid]
            if sector_values.size >= 10:
                expected_target[sector_mask] = float(
                    np.quantile(sector_values, 0.5, method="linear")
                )
                expected_source[sector_mask] = "SECTOR"
                expected_population[sector_mask] = sector_values.size
        expected_shrunk = (
            nobs[idx] / (nobs[idx] + 60.0) * raw[idx]
            + 60.0 / (nobs[idx] + 60.0) * expected_target
        )
        shrinkage_mismatches += int(
            np.sum(~_nullable_equal(target_observed[idx], expected_target))
            + np.sum(source_observed[idx] != expected_source)
            + np.sum(population_observed[idx] != expected_population)
            + np.sum(~_nullable_equal(shrunk_observed[idx], expected_shrunk))
        )

        omega = covariance_by_day.get(int(day))
        _require(omega is not None, f"Missing independently computed covariance for {day}")
        pre = np.einsum("ij,jk,ik->i", betas[idx], omega, betas[idx]) + expected_shrunk
        ridge = max(1e-8, 1e-6 * float(np.median(pre)))
        ridge_mismatches += int(
            np.sum(~_nullable_equal(ridge_observed[idx], np.full(idx.size, ridge)))
            + np.sum(~_nullable_equal(specific_daily[idx], expected_shrunk + ridge))
            + np.sum(~_nullable_equal(security_pre[idx], pre))
            + np.sum(~_nullable_equal(security_daily[idx], pre + ridge))
        )
        monthly_mismatches += int(
            np.sum(~_nullable_equal(specific_monthly[idx], 21.0 * (expected_shrunk + ridge)))
            + np.sum(~_nullable_equal(security_monthly[idx], 21.0 * (pre + ridge)))
        )

        complete = (
            np.isfinite(spread[idx])
            & (spread[idx] >= 0.0)
            & np.isfinite(adv[idx])
            & (adv[idx] > 0.0)
            & np.isfinite(impact[idx])
            & (impact[idx] >= 0.0)
        )
        expected_status: list[str] = []
        for spread_value, adv_value, impact_value in zip(
            spread[idx], adv[idx], impact[idx], strict=True
        ):
            missing: list[str] = []
            if not math.isfinite(float(spread_value)) or spread_value < 0.0:
                missing.append("MISSING_SPREAD")
            if not math.isfinite(float(adv_value)) or adv_value <= 0.0:
                missing.append("MISSING_ADV")
            if not math.isfinite(float(impact_value)) or impact_value < 0.0:
                missing.append("MISSING_IMPACT_VOLATILITY")
            expected_status.append("PRESENT" if not missing else "|".join(missing))
        permission_mismatches += int(
            np.sum(new_long[idx] != complete)
            + np.sum(new_short[idx] != (short[idx] & complete))
            + np.sum(cost_status[idx] != np.asarray(expected_status, dtype=object))
        )
        finite_impact = impact[idx][np.isfinite(impact[idx]) & (impact[idx] >= 0.0)]
        p95 = float(np.quantile(finite_impact, 0.95, method="linear"))
        expected_exit = np.where(np.isfinite(impact[idx]) & (impact[idx] >= 0.0), impact[idx], p95)
        expected_exit_source = np.where(
            np.isfinite(impact[idx]) & (impact[idx] >= 0.0),
            "SECURITY",
            "CROSS_SECTION_P95",
        )
        exit_mismatches += int(
            np.sum(~_nullable_equal(exit_vol[idx], expected_exit))
            + np.sum(exit_source[idx] != expected_exit_source)
        )

        short_local = np.flatnonzero(short[idx])
        if short_local.size:
            absolute = idx[short_local]
            expected_q = independent_liquidity_quintiles(
                [int(value) for value in permnos[absolute]],
                [None if not math.isfinite(float(value)) else float(value) for value in adv[absolute]],
            )
            expected_stress = np.where(expected_q == 1, 0.10, 0.03)
            borrow_mismatches += int(
                np.sum(quintile[absolute] != expected_q)
                + np.sum(~_nullable_equal(primary_borrow[absolute], np.full(absolute.size, 0.01)))
                + np.sum(~_nullable_equal(stress_borrow[absolute], expected_stress))
            )
        nonshort = idx[~short[idx]]
        borrow_mismatches += int(
            np.sum(quintile[nonshort] != 0)
            + np.sum(np.isfinite(primary_borrow[nonshort]))
            + np.sum(np.isfinite(stress_borrow[nonshort]))
        )

    counters = {
        "full_shrinkage_formula_mismatches": shrinkage_mismatches,
        "full_ridge_formula_mismatches": ridge_mismatches,
        "full_monthly_scale_mismatches": monthly_mismatches,
        "full_cost_permission_mismatches": permission_mismatches,
        "full_exit_volatility_mismatches": exit_mismatches,
        "full_borrow_quintile_rate_mismatches": borrow_mismatches,
    }
    differences = [
        Difference(
            key,
            0,
            value,
            "Full vectorized independent check of every processed security-signal row.",
        )
        for key, value in counters.items()
        if value
    ]
    return {"full_vector_rows_checked": processed.num_rows, **counters}, differences


def _write_table(path: Path, table: pa.Table, policy: Mapping[str, Any]) -> dict[str, Any]:
    pq.write_table(
        table,
        path,
        row_group_size=int(policy["build"]["row_group_size"]),
        compression=str(policy["build"]["compression"]),
        use_dictionary=bool(policy["build"]["dictionary_encode_strings"]),
        version="2.6",
        write_statistics=True,
        data_page_version="1.0",
    )
    return _file_receipt(path, parquet=True)


def _byte_replay(
    artifacts: AuthenticatedPrimary,
    *,
    batch_rows: int,
    progress: Callable[[str], None],
) -> tuple[dict[str, Any], list[Difference], Any]:
    result = materialize_risk_cost(
        artifacts.inputs,
        artifacts.rules,
        batch_rows=batch_rows,
        progress=progress,
    )
    table_map = {
        "development_risk_cost_audit.parquet": result.audit,
        "development_risk_cost_inputs.parquet": result.processed,
        "development_factor_covariance.parquet": result.factor_covariance,
        "development_risk_cost_month_summary.parquet": result.month_summary,
        "risk_cost_timing_checks.parquet": result.timing,
        "risk_cost_coverage.parquet": result.coverage,
    }
    json_map = {
        "model_parameters.json": result.model_parameters,
        "quality_summary.json": result.metrics,
    }
    differences: list[Difference] = []
    files_checked = 0
    with tempfile.TemporaryDirectory(prefix="stage8-independent-replay-") as temporary:
        root = Path(temporary)
        for location in ("interim", "processed", "quality"):
            receipts: dict[str, dict[str, Any]] = {}
            expected_files = tuple(
                artifacts.policy["outputs"][location]["required_files"]
            )
            declared = artifacts.manifest["outputs"][location]["files"]
            for name in expected_files:
                if name == "_SUCCESS.json":
                    continue
                path = root / f"{location}-{name}"
                if name in table_map:
                    receipt = _write_table(path, table_map[name], artifacts.policy)
                else:
                    payload = canonical_json_bytes(json_map[name])
                    path.write_bytes(payload)
                    receipt = _file_receipt(path, parquet=False)
                receipt["path"] = name
                receipts[name] = receipt
                files_checked += 1
                expected = dict(declared[name])
                if receipt != expected:
                    differences.append(
                        Difference(
                            f"byte_replay:{location}/{name}",
                            expected,
                            receipt,
                            "Deterministic full materialization replay did not reproduce the primary payload receipt.",
                        )
                    )
            success = {
                "schema_version": 1,
                "contract_id": artifacts.policy["contract_id"],
                "policy_id": artifacts.policy["policy_id"],
                "build_id": artifacts.build_id,
                "content_fingerprint": artifacts.manifest["bindings"]["content_fingerprint"],
                "location": location,
                "status": f"PRIMARY_{location.upper()}_COMPLETE",
                "files": {name: receipts[name] for name in sorted(receipts)},
            }
            success_hash = _sha256_bytes(canonical_json_bytes(success))
            files_checked += 1
            expected_success = artifacts.manifest["outputs"][location]["success_sha256"]
            if success_hash != expected_success:
                differences.append(
                    Difference(
                        f"byte_replay:{location}/_SUCCESS.json",
                        expected_success,
                        success_hash,
                        "Deterministic success receipt replay differs.",
                    )
                )
    return {
        "full_materialization_replayed": True,
        "primary_payload_files_replayed": files_checked,
        "primary_payload_hash_mismatches": len(differences),
    }, differences, result


def _verification_code_hashes() -> dict[str, str]:
    return {name: sha256_file(project_path(name)) for name in _VERIFICATION_CODE_PATHS}


def _differences_table(differences: Sequence[Difference]) -> pa.Table:
    schema = pa.schema(
        [
            pa.field("code", pa.string(), nullable=False),
            pa.field("expected_json", pa.string(), nullable=False),
            pa.field("observed_json", pa.string(), nullable=False),
            pa.field("details", pa.string(), nullable=False),
        ]
    )
    rows = [
        {
            "code": item.code,
            "expected_json": json.dumps(
                _json_safe(item.expected), sort_keys=True, separators=(",", ":"), allow_nan=False
            ),
            "observed_json": json.dumps(
                _json_safe(item.observed), sort_keys=True, separators=(",", ":"), allow_nan=False
            ),
            "details": item.details,
        }
        for item in differences
    ]
    return pa.Table.from_pylist(rows, schema=schema)


def _parquet_bytes(table: pa.Table) -> bytes:
    sink = pa.BufferOutputStream()
    pq.write_table(
        table,
        sink,
        row_group_size=1024,
        compression="zstd",
        use_dictionary=True,
        version="2.6",
        write_statistics=True,
        data_page_version="1.0",
    )
    return sink.getvalue().to_pybytes()


def _reauthenticate(artifacts: AuthenticatedPrimary) -> None:
    observed = authenticate_primary(
        policy_path=artifacts.policy_path,
        primary_pointer_path=artifacts.primary_pointer_path,
    )
    _require(
        observed.primary_pointer_sha256 == artifacts.primary_pointer_sha256
        and observed.primary_manifest_sha256 == artifacts.primary_manifest_sha256
        and observed.manifest == artifacts.manifest
        and observed.output_hashes == artifacts.output_hashes
        and observed.stat_identities == artifacts.stat_identities,
        "Primary artifact graph changed during independent verification",
    )


def _publish_verified(
    artifacts: AuthenticatedPrimary,
    checks: Mapping[str, Any],
    differences: Sequence[Difference],
) -> tuple[str, Path, Path]:
    _require(not differences, "Cannot publish Stage 8 authority with differences")
    code_hashes = _verification_code_hashes()
    differences_payload = _parquet_bytes(_differences_table(differences))
    differences_hash = _sha256_bytes(differences_payload)
    identity = {
        "build_id": artifacts.build_id,
        "primary_manifest_sha256": artifacts.primary_manifest_sha256,
        "primary_pointer_sha256": artifacts.primary_pointer_sha256,
        "verification_code_hashes": code_hashes,
        "checks": _json_safe(checks),
        "differences_sha256": differences_hash,
        "output_hashes": artifacts.output_hashes,
    }
    verification_id = "verify-" + _sha256_bytes(canonical_json_bytes(identity))[:16]
    _require(_VERIFY_ID.fullmatch(verification_id) is not None, "Invalid verification ID")
    independent_check = {
        "schema_version": 1,
        "contract_id": artifacts.policy["contract_id"],
        "policy_id": artifacts.policy["policy_id"],
        "build_id": artifacts.build_id,
        "verification_id": verification_id,
        "status": _VERIFIED_STATUS,
        "primary_manifest_sha256": artifacts.primary_manifest_sha256,
        "primary_pointer_sha256": artifacts.primary_pointer_sha256,
        "verification_code_hashes": code_hashes,
        "checks": _json_safe(checks),
        "difference_count": 0,
        "differences_sha256": differences_hash,
        "authorizations": _VERIFIED_AUTHORIZATIONS,
    }
    check_payload = canonical_json_bytes(independent_check)
    check_hash = _sha256_bytes(check_payload)
    pointer = {
        "schema_version": 1,
        "contract_id": artifacts.policy["contract_id"],
        "policy_id": artifacts.policy["policy_id"],
        "build_id": artifacts.build_id,
        "verification_id": verification_id,
        "status": _VERIFIED_STATUS,
        "primary_manifest_sha256": artifacts.primary_manifest_sha256,
        "independent_check_sha256": check_hash,
        "differences_sha256": differences_hash,
        "authorizations": _VERIFIED_AUTHORIZATIONS,
    }
    artifact_root = project_path(artifacts.policy["publication"]["artifact_root"])
    destination = (
        artifact_root
        / str(artifacts.policy["publication"]["verification_namespace"])
        / artifacts.build_id
        / verification_id
    )

    def gate() -> None:
        _reauthenticate(artifacts)
        _require(
            _verification_code_hashes() == code_hashes,
            "Verification code changed during publication",
        )

    payloads = {
        "differences.parquet": differences_payload,
        "independent_check.json": check_payload,
    }
    expected_hashes = {name: _sha256_bytes(value) for name, value in payloads.items()}
    gate()
    try:
        with SecureStagingDirectory.create(
            trusted_root=artifact_root,
            destination=destination,
            expected_files=tuple(payloads),
            label="Stage 8 independent verification",
        ) as stage:
            for name, payload in payloads.items():
                with stage.open_exclusive_binary(name) as stream:
                    stream.write(payload)
            snapshot = stage.snapshot(expected_hashes=expected_hashes)
            stage.publish(expected_hashes=snapshot.hashes)
    except SecurePublicationError as exc:
        raise Stage8VerificationError("Cannot publish Stage 8 verification bundle") from exc
    gate()
    pointer_path = project_path(
        artifacts.policy["publication"]["latest_verified_pointer"]
    )
    try:
        replace_pointer_last(
            pointer_path,
            canonical_json_bytes(pointer),
            trusted_root=artifact_root,
            pre_replace_gate=gate,
            post_replace_gate=gate,
            label="Stage 8 latest verified pointer",
        )
    except SecurePublicationError as exc:
        raise Stage8VerificationError("Cannot replace Stage 8 verified pointer") from exc
    return verification_id, destination, pointer_path


def verify_stage8_risk_cost(
    *,
    policy_path: Path = DEFAULT_POLICY,
    primary_pointer_path: Path | None = None,
    batch_rows: int = 262_144,
    publish: bool = True,
    progress: Callable[[str], None] | None = None,
) -> VerificationResult:
    """Replay, independently test, and optionally publish Stage 8 authority."""

    _require(
        isinstance(batch_rows, int) and not isinstance(batch_rows, bool) and batch_rows > 0,
        "Verification batch size must be a positive integer",
    )
    report = progress or (lambda _message: None)
    artifacts = authenticate_primary(
        policy_path=policy_path,
        primary_pointer_path=primary_pointer_path,
    )
    checks: dict[str, Any] = {
        "primary_artifact_graph_authenticated": True,
        "primary_output_locations_authenticated": 3,
        "primary_output_files_authenticated": sum(
            len(value) for value in artifacts.output_paths.values()
        ),
        "input_tables_authenticated": len(artifacts.inputs.input_receipts),
        "primary_hard_gates_authenticated": len(artifacts.policy["quality_gates"]),
    }
    differences: list[Difference] = []

    report("full deterministic primary materialization replay")
    replay_checks, replay_differences, replay = _byte_replay(
        artifacts, batch_rows=batch_rows, progress=report
    )
    checks.update(replay_checks)
    differences.extend(replay_differences)

    report("independent factor covariance recomputation")
    context = _factor_context(artifacts.inputs)
    factor_checks, factor_differences, covariance_by_day = _factor_scientific_check(
        artifacts, context
    )
    checks.update(factor_checks)
    differences.extend(factor_differences)

    # Use the replayed table only after its serialized bytes have been proved
    # identical to the authenticated primary payload.
    processed = replay.processed
    report("independent stratified EWLS and impact-volatility audit")
    sample_checks, sample_differences = _stratified_security_check(
        artifacts,
        processed,
        replay.audit,
        context,
        batch_rows=batch_rows,
        progress=report,
    )
    checks.update(sample_checks)
    differences.extend(sample_differences)

    report("full vectorized shrinkage, ridge, cost, and borrow checks")
    vector_checks, vector_differences = _full_vector_checks(
        processed, covariance_by_day
    )
    checks.update(vector_checks)
    differences.extend(vector_differences)
    checks["scientific_check_difference_count"] = len(differences)
    checks["verification_code_hashes"] = _verification_code_hashes()

    verification_id: str | None = None
    verification_path: Path | None = None
    verified_pointer_path: Path | None = None
    if not differences and publish:
        report("immutable verification publication and pointer-last authority")
        verification_id, verification_path, verified_pointer_path = _publish_verified(
            artifacts, checks, differences
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
