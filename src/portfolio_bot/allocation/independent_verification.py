"""Independent, synthetic-only verification for roadmap Stage 10.

The primary Stage 10 bundle contains no empirical portfolio payload.  This
verifier holds that exact two-file bundle open, reauthenticates every bound
dependency, replays the synthetic evidence, performs a separate NumPy
transcription of the allocator economics and constraints, and records the
focused and repository-wide regression results.  A passing pointer establishes
only the technical precondition for Stage 11; explicit user approval is still
required before any 2015--2019 row, portfolio, trade, or P&L is created.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from portfolio_bot.allocation.contract import (
    DEFAULT_POLICY,
    PROJECT_ROOT,
    Stage10ContractReceipt,
    authenticate_stage10_contract,
    canonical_json_bytes,
    load_policy,
    sha256_file,
)
from portfolio_bot.allocation.decision_focused import (
    PRODUCTION_TRAINING_RUNTIME_CERTIFIED,
)
from portfolio_bot.allocation.evidence import (
    build_synthetic_allocator_evidence,
    synthetic_evidence_sha256,
)
from portfolio_bot.allocation.fixtures import balanced_synthetic_problem
from portfolio_bot.allocation.publication import (
    DEPENDENCY_CODE_PATHS,
    MANIFEST_NAME,
    PRIMARY_FILES,
    SUCCESS_NAME,
)
from portfolio_bot.allocation.solver import solve_hybrid
from portfolio_bot.data.secure_publication import (
    AuthenticatedBundleGuard,
    AuthenticatedPointerGuard,
    PublishedBundle,
    SecureStagingDirectory,
    replace_pointer_last,
)


VERIFICATION_CHECKS_NAME = "verification_checks.parquet"
VERIFICATION_REPORT_NAME = "verification_report.json"
VERIFICATION_FILES = (
    VERIFICATION_CHECKS_NAME,
    VERIFICATION_REPORT_NAME,
    SUCCESS_NAME,
)
VERIFICATION_CODE_PATHS = (
    "src/portfolio_bot/allocation/independent_verification.py",
    "scripts/verify_stage_10_allocators.py",
)
VERIFY_ID_PATTERN = re.compile(r"verify-[0-9a-f]{16}\Z")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
FORWARD_TOLERANCE = 5e-8

VERIFIED_AUTHORIZATIONS = {
    "independent_verification_completed": True,
    "stage_11_validation_technical_gate_passed": True,
    "stage_11_requires_explicit_user_approval": True,
    "stage_11_execution_authorized_now": False,
    "stage_11_execution_started": False,
    "validation_2015_2019_access_authorized_now": False,
    "stage_12_or_later_authorized": False,
    "final_2020_2025_access_authorized": False,
    "real_portfolio_weights_trades_or_pnl_authorized_now": False,
}


class AllocatorVerificationError(RuntimeError):
    """Raised when Stage 10 cannot be independently authenticated."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AllocatorVerificationError(message)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AllocatorVerificationError(f"invalid {label} JSON") from exc
    _require(isinstance(value, dict), f"{label} is not a JSON object")
    return value


def _safe_relative(value: object, *, label: str) -> Path:
    path = Path(str(value))
    _require(
        bool(path.parts) and not path.is_absolute() and ".." not in path.parts,
        f"unsafe {label} path",
    )
    return path


@dataclass(frozen=True)
class RegressionReceipt:
    """Hash-only receipt for one exact pytest subprocess."""

    label: str
    command: tuple[str, ...]
    exit_code: int
    duration_seconds: float
    stdout_sha256: str
    stderr_sha256: str
    source: str

    def __post_init__(self) -> None:
        _require(
            self.label in {"focused_stage10", "full_repository"},
            "unknown test receipt label",
        )
        _require(
            bool(self.command) and all(str(value) for value in self.command),
            "empty test command",
        )
        _require(
            not isinstance(self.exit_code, bool) and isinstance(self.exit_code, int),
            "invalid test exit code",
        )
        _require(
            math.isfinite(float(self.duration_seconds))
            and self.duration_seconds >= 0.0,
            "invalid test duration",
        )
        _require(
            SHA256_PATTERN.fullmatch(self.stdout_sha256) is not None,
            "malformed stdout hash",
        )
        _require(
            SHA256_PATTERN.fullmatch(self.stderr_sha256) is not None,
            "malformed stderr hash",
        )
        _require(
            self.source in {"executed", "injected_synthetic_test"},
            "invalid test receipt source",
        )

    @property
    def passed(self) -> bool:
        return self.exit_code == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "command": list(self.command),
            "exit_code": self.exit_code,
            "duration_seconds": float(self.duration_seconds),
            "stdout_sha256": self.stdout_sha256,
            "stderr_sha256": self.stderr_sha256,
            "source": self.source,
            "passed": self.passed,
        }


@dataclass
class AuthenticatedPrimary:
    """Retained primary pointer and exact immutable two-file bundle."""

    policy: Mapping[str, Any]
    receipt: Stage10ContractReceipt
    pointer_path: Path
    pointer_payload: bytes
    pointer_sha256: str
    pointer: Mapping[str, Any]
    manifest_payload: bytes
    manifest_sha256: str
    manifest: Mapping[str, Any]
    pointer_guard: AuthenticatedPointerGuard
    manifest_guard: AuthenticatedPointerGuard
    success_guard: AuthenticatedPointerGuard
    bundle_guard: AuthenticatedBundleGuard

    @property
    def build_id(self) -> str:
        return str(self.manifest["build_id"])

    def gate(self) -> None:
        self.pointer_guard.gate()
        self.manifest_guard.gate()
        self.success_guard.gate()
        self.bundle_guard.gate()
        current = authenticate_stage10_contract()
        _require(
            current.sha256 == self.receipt.sha256,
            "Stage 10 contract changed during verification",
        )
        for group in ("code_hashes", "configuration_hashes"):
            records = self.manifest["bindings"][group]
            for name, expected in records.items():
                path = PROJECT_ROOT / str(name)
                _require(
                    path.is_file()
                    and not path.is_symlink()
                    and sha256_file(path) == str(expected),
                    f"bound dependency changed: {name}",
                )


def _primary_pointer_validator(payload: bytes) -> None:
    pointer = _json(payload, label="Stage 10 primary pointer")
    _require(
        pointer.get("status") == "PRIMARY_PASSED_PENDING_INDEPENDENT_VERIFICATION",
        "primary pointer status differs",
    )
    authorizations = pointer.get("authorizations")
    _require(isinstance(authorizations, Mapping), "primary authorizations are missing")
    _require(
        authorizations.get("independent_verification_authorized") is True,
        "primary does not authorize verification",
    )
    for name in (
        "stage_11_validation_authorized",
        "stage_11_or_later_authorized",
        "stage_12_or_later_authorized",
        "validation_or_final_test_access_authorized",
        "real_portfolio_weights_or_trades_authorized",
        "real_pnl_or_performance_authorized",
    ):
        _require(
            authorizations.get(name) is False, f"primary authority escaped at {name}"
        )


@contextlib.contextmanager
def authenticate_primary_allocator(
    *,
    policy_path: Path = DEFAULT_POLICY,
    project_root: Path = PROJECT_ROOT,
    primary_pointer_path: Path | None = None,
    trusted_root: Path | None = None,
) -> Iterator[AuthenticatedPrimary]:
    """Hold and revalidate the exact Stage 10 primary publication graph."""

    policy = load_policy(policy_path)
    outputs = policy["outputs"]
    root = (
        Path(trusted_root).absolute()
        if trusted_root is not None
        else project_root.absolute()
    )
    pointer_path = (
        Path(primary_pointer_path).absolute()
        if primary_pointer_path is not None
        else (project_root / str(outputs["latest_primary_pointer"])).absolute()
    )
    receipt = authenticate_stage10_contract(policy_path, project_root=project_root)
    with contextlib.ExitStack() as stack:
        pointer_guard = stack.enter_context(
            AuthenticatedPointerGuard.open(
                trusted_root=root,
                path=pointer_path,
                payload_validator=_primary_pointer_validator,
                label="Stage 10 primary allocator pointer",
            )
        )
        pointer = _json(pointer_guard.payload, label="Stage 10 primary pointer")
        manifest_relative = _safe_relative(
            pointer.get("primary_manifest_path"),
            label="primary manifest",
        )
        manifest_path = root / manifest_relative
        manifest_guard = stack.enter_context(
            AuthenticatedPointerGuard.open(
                trusted_root=root,
                path=manifest_path,
                expected_sha256=str(pointer.get("primary_manifest_sha256")),
                label="Stage 10 primary allocator manifest",
            )
        )
        manifest = _json(manifest_guard.payload, label="Stage 10 primary manifest")
        _require(
            manifest.get("build_id") == pointer.get("build_id"),
            "primary build binding differs",
        )
        _require(
            manifest.get("status") == pointer.get("status"),
            "primary status binding differs",
        )
        success_guard = stack.enter_context(
            AuthenticatedPointerGuard.open(
                trusted_root=root,
                path=manifest_path.parent / SUCCESS_NAME,
                label="Stage 10 primary allocator completion receipt",
            )
        )
        success = _json(
            success_guard.payload, label="Stage 10 primary completion receipt"
        )
        _require(
            success.get("status") == "PRIMARY_MANIFEST_COMPLETE",
            "primary completion status differs",
        )
        _require(
            success.get("build_id") == manifest.get("build_id"),
            "primary completion build differs",
        )
        _require(
            success.get("manifest_sha256") == manifest_guard.payload_sha256,
            "primary completion hash differs",
        )
        _require(
            tuple(success.get("closed_files", ())) == PRIMARY_FILES,
            "primary completion closed set differs",
        )
        bundle_guard = stack.enter_context(
            AuthenticatedBundleGuard.open(
                trusted_root=root,
                path=manifest_path.parent,
                expected_files=PRIMARY_FILES,
                expected_hashes={
                    MANIFEST_NAME: manifest_guard.payload_sha256,
                    SUCCESS_NAME: success_guard.payload_sha256,
                },
                label="Stage 10 primary allocator bundle",
            )
        )
        bindings = manifest.get("bindings")
        _require(isinstance(bindings, Mapping), "primary bindings are missing")
        code_hashes = bindings.get("code_hashes")
        configuration_hashes = bindings.get("configuration_hashes")
        _require(isinstance(code_hashes, Mapping), "primary code hashes are missing")
        _require(
            isinstance(configuration_hashes, Mapping),
            "primary configuration hashes are missing",
        )
        _require(
            set(code_hashes) == set(DEPENDENCY_CODE_PATHS),
            "primary dependency closed set differs",
        )
        for records in (code_hashes, configuration_hashes):
            for name, expected in records.items():
                path = project_root / str(name)
                _require(
                    path.is_file()
                    and not path.is_symlink()
                    and sha256_file(path) == str(expected),
                    f"primary dependency hash differs: {name}",
                )
        _require(
            manifest.get("contract_receipt") == receipt.to_dict(),
            "primary contract receipt differs",
        )
        _require(
            bindings.get("contract_receipt_sha256") == receipt.sha256,
            "primary contract hash differs",
        )
        evidence = manifest.get("synthetic_evidence")
        _require(isinstance(evidence, Mapping), "primary synthetic evidence is missing")
        _require(
            bindings.get("synthetic_evidence_sha256")
            == synthetic_evidence_sha256(evidence),
            "primary synthetic evidence hash differs",
        )
        scope = evidence.get("scope")
        _require(isinstance(scope, Mapping), "primary synthetic scope is missing")
        for name in (
            "real_data_payload_rows_read",
            "validation_or_final_rows_read",
            "real_portfolio_weights_created",
            "real_trades_created",
            "real_pnl_or_performance_rows_created",
        ):
            _require(scope.get(name) == 0, f"primary empirical counter differs: {name}")
        primary = AuthenticatedPrimary(
            policy=policy,
            receipt=receipt,
            pointer_path=pointer_path,
            pointer_payload=pointer_guard.payload,
            pointer_sha256=pointer_guard.payload_sha256,
            pointer=pointer,
            manifest_payload=manifest_guard.payload,
            manifest_sha256=manifest_guard.payload_sha256,
            manifest=manifest,
            pointer_guard=pointer_guard,
            manifest_guard=manifest_guard,
            success_guard=success_guard,
            bundle_guard=bundle_guard,
        )
        primary.gate()
        yield primary
        primary.gate()


def independent_allocator_math() -> dict[str, Any]:
    """Recompute one synthetic solve without using diagnostic/cost helpers."""

    data = balanced_synthetic_problem()
    result = solve_hybrid(data)
    weights = np.asarray(result.weights, dtype=np.float64)
    previous = np.asarray(data.previous_weight, dtype=np.float64)
    delta = weights - previous
    absolute_delta = np.abs(delta)
    transaction = float(
        np.sum(
            absolute_delta
            * (
                data.full_spread / 2.0
                + 0.0001
                + 0.10
                * data.impact_volatility_daily
                * np.sqrt(absolute_delta * data.nav_dollars / data.adv20_usd)
            )
        )
    )
    borrow = float(np.sum(np.maximum(-weights, 0.0) * data.borrow_rate_annual / 12.0))
    gross = float(np.sum(np.abs(weights)))
    financing = max(gross - 1.0, 0.0) * 0.005 / 12.0
    long_gross = float(np.sum(weights[data.long_mask]))
    short_gross = float(-np.sum(weights[data.short_mask]))
    net = long_gross - short_gross
    turnover = 0.5 * float(np.sum(absolute_delta))
    factor = data.factor_betas.T @ weights
    ordinary_monthly_variance = float(
        factor @ data.factor_covariance_monthly @ factor
        + np.dot(data.specific_variance_monthly, weights**2)
    )
    augmented_annual_variance = 12.0 * (
        ordinary_monthly_variance
        + 4.0 * data.market_variance_monthly * float(factor[0]) ** 2
    )

    checks = {
        "result_is_ordinary": result.mode == "ordinary",
        "transaction_cost_matches_manual": math.isclose(
            result.transaction_cost, transaction, rel_tol=1e-11, abs_tol=1e-13
        ),
        "borrow_cost_matches_manual": math.isclose(
            result.borrow_cost, borrow, rel_tol=1e-11, abs_tol=1e-13
        ),
        "financing_cost_matches_manual": math.isclose(
            result.financing_cost, financing, rel_tol=1e-11, abs_tol=1e-13
        ),
        "total_cost_matches_manual": math.isclose(
            result.total_cost,
            transaction + borrow + financing,
            rel_tol=1e-11,
            abs_tol=1e-13,
        ),
        "fixed_signs_hold": bool(
            np.all(weights[data.long_mask] >= -FORWARD_TOLERANCE)
            and np.all(weights[data.short_mask] <= FORWARD_TOLERANCE)
        ),
        "position_caps_hold": bool(
            np.all(weights[data.long_mask] <= 0.0100 + FORWARD_TOLERANCE)
            and np.all(-weights[data.short_mask] <= 0.0075 + FORWARD_TOLERANCE)
        ),
        "gross_limit_holds": gross <= 2.0 + FORWARD_TOLERANCE,
        "net_nav_limit_holds": abs(net) <= 0.10 + FORWARD_TOLERANCE,
        "dynamic_net_limit_holds": abs(net) <= 0.05 * gross + FORWARD_TOLERANCE,
        "turnover_limit_holds": turnover <= 0.50 + FORWARD_TOLERANCE,
        "trade_capacity_holds": bool(
            np.all(absolute_delta <= data.trade_capacity_weight + FORWARD_TOLERANCE)
        ),
        "adv_capacity_holds": bool(
            np.all(
                absolute_delta
                <= 0.10 * data.adv20_usd / data.nav_dollars + FORWARD_TOLERANCE
            )
        ),
        "factor_limits_hold": bool(np.all(np.abs(factor) <= 0.10 + FORWARD_TOLERANCE)),
        "augmented_risk_holds": augmented_annual_variance
        <= 0.10**2 + FORWARD_TOLERANCE,
        "risk_matches_solver_diagnostic": math.isclose(
            result.diagnostics.augmented_annual_variance,
            augmented_annual_variance,
            rel_tol=1e-11,
            abs_tol=1e-13,
        ),
    }
    for issuer in np.unique(data.permco):
        mask = data.permco == issuer
        checks[f"issuer_cap:{int(issuer)}"] = bool(
            np.sum(weights[mask & data.long_mask]) <= 0.0100 + FORWARD_TOLERANCE
            and -np.sum(weights[mask & data.short_mask]) <= 0.0075 + FORWARD_TOLERANCE
        )
    for sector in np.unique(data.sector):
        values = weights[data.sector == sector]
        checks[f"sector:{sector}"] = bool(
            abs(float(np.sum(values))) <= 0.05 + FORWARD_TOLERANCE
            and float(np.sum(np.abs(values))) <= 0.30 + FORWARD_TOLERANCE
        )
    for country in np.unique(data.country):
        if country == "USA":
            continue
        checks[f"country:{country}"] = bool(
            abs(float(np.sum(weights[data.country == country])))
            <= 0.05 + FORWARD_TOLERANCE
        )
    checks["aggregate_foreign"] = bool(
        abs(float(np.sum(weights[data.country != "USA"]))) <= 0.10 + FORWARD_TOLERANCE
    )
    _require(
        all(value is True for value in checks.values()),
        "independent allocator math differs",
    )
    return {
        "checks": checks,
        "weights_sha256": _sha256_bytes(weights.astype("<f8", copy=False).tobytes()),
        "manual_transaction_cost": transaction,
        "manual_borrow_cost": borrow,
        "manual_financing_cost": financing,
        "manual_augmented_annual_variance": augmented_annual_variance,
        "manual_gross": gross,
        "manual_net": net,
        "manual_turnover": turnover,
    }


def _run_pytest(
    label: str, command: Sequence[str], *, project_root: Path
) -> RegressionReceipt:
    started = time.monotonic()
    completed = subprocess.run(
        tuple(command),
        cwd=project_root,
        capture_output=True,
        check=False,
    )
    duration = time.monotonic() - started
    return RegressionReceipt(
        label=label,
        command=tuple(str(value) for value in command),
        exit_code=int(completed.returncode),
        duration_seconds=duration,
        stdout_sha256=_sha256_bytes(completed.stdout),
        stderr_sha256=_sha256_bytes(completed.stderr),
        source="executed",
    )


def run_regression_tests(
    *, project_root: Path = PROJECT_ROOT
) -> tuple[RegressionReceipt, RegressionReceipt]:
    """Run the focused Stage 10 suite and the full repository exactly once."""

    python = project_root / ".venv/bin/python"
    executable = str(python if python.is_file() else Path(sys.executable))
    focused_files = sorted((project_root / "tests/unit").glob("test_stage10_*.py"))
    focused_files.extend(
        sorted((project_root / "tests/integration").glob("test_stage10_*.py"))
    )
    _require(bool(focused_files), "no focused Stage 10 tests were found")
    focused_command = (
        executable,
        "-m",
        "pytest",
        "-q",
        *(str(path.relative_to(project_root)) for path in focused_files),
    )
    full_command = (executable, "-m", "pytest", "-q")
    focused = _run_pytest("focused_stage10", focused_command, project_root=project_root)
    _require(focused.passed, "focused Stage 10 regression failed")
    full = _run_pytest("full_repository", full_command, project_root=project_root)
    _require(full.passed, "full repository regression failed")
    return focused, full


def verification_checks_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("check_name", pa.string(), nullable=False),
            pa.field("status", pa.string(), nullable=False),
            pa.field("observed_json", pa.string(), nullable=False),
            pa.field("detail", pa.string(), nullable=False),
        ],
        metadata={b"contract": b"phase04-allocation-v1", b"stage": b"10"},
    )


def _checks_table(checks: Mapping[str, Any]) -> pa.Table:
    rows = [
        {
            "check_name": str(name),
            "status": "PASS",
            "observed_json": json.dumps(
                value, sort_keys=True, separators=(",", ":"), allow_nan=False
            ),
            "detail": "Independent Stage 10 verification passed.",
        }
        for name, value in sorted(checks.items())
    ]
    return pa.Table.from_pylist(rows, schema=verification_checks_schema())


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
    )
    return sink.getvalue().to_pybytes()


def _verification_code_hashes(*, project_root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name in VERIFICATION_CODE_PATHS:
        path = project_root / name
        _require(
            path.is_file() and not path.is_symlink(),
            f"missing verifier dependency: {name}",
        )
        hashes[name] = sha256_file(path)
    return hashes


@dataclass(frozen=True)
class AllocatorVerificationResult:
    passed: bool
    build_id: str
    verification_id: str | None
    checks: Mapping[str, Any]
    verification_path: Path | None
    verified_pointer_path: Path | None
    bundle: PublishedBundle | None


def _publish_verification(
    primary: AuthenticatedPrimary,
    checks: Mapping[str, Any],
    focused: RegressionReceipt,
    full: RegressionReceipt,
    *,
    project_root: Path,
    artifact_root: Path | None,
    verified_pointer_path: Path | None,
    trusted_root: Path | None,
) -> tuple[str, Path, Path, PublishedBundle]:
    code_hashes = _verification_code_hashes(project_root=project_root)
    checks_payload = _parquet_bytes(_checks_table(checks))
    identity = {
        "build_id": primary.build_id,
        "primary_pointer_sha256": primary.pointer_sha256,
        "primary_manifest_sha256": primary.manifest_sha256,
        "verification_code_hashes": code_hashes,
        "focused_test_receipt": focused.to_dict(),
        "full_test_receipt": full.to_dict(),
        "verification_checks_sha256": _sha256_bytes(checks_payload),
    }
    verification_id = "verify-" + _sha256_bytes(canonical_json_bytes(identity))[:16]
    _require(
        VERIFY_ID_PATTERN.fullmatch(verification_id) is not None,
        "malformed verification ID",
    )
    report = {
        "schema_version": 1,
        "contract_id": primary.receipt.contract_id,
        "policy_id": primary.receipt.policy_id,
        "rules_id": primary.receipt.rules_id,
        "build_id": primary.build_id,
        "verification_id": verification_id,
        "status": "VERIFIED_PASSED",
        "primary_pointer_sha256": primary.pointer_sha256,
        "primary_manifest_sha256": primary.manifest_sha256,
        "verification_code_hashes": code_hashes,
        "checks": dict(checks),
        "focused_test_receipt": focused.to_dict(),
        "full_test_receipt": full.to_dict(),
        "verification_checks_sha256": _sha256_bytes(checks_payload),
        "difference_count": 0,
        "authorizations": VERIFIED_AUTHORIZATIONS,
        "limitations": {
            "decision_focused_empirical_production_training": "INACTIVE_PENDING_STAGE11_DYNAMIC_ADAPTER_AND_CERTIFIED_REUSABLE_PROJECTION"
        },
    }
    report_payload = canonical_json_bytes(report)
    success_payload = canonical_json_bytes(
        {
            "schema_version": 1,
            "build_id": primary.build_id,
            "verification_id": verification_id,
            "status": "VERIFICATION_BUNDLE_COMPLETE",
            "verification_report_sha256": _sha256_bytes(report_payload),
            "verification_checks_sha256": _sha256_bytes(checks_payload),
            "closed_files": list(VERIFICATION_FILES),
        }
    )
    payloads = {
        VERIFICATION_CHECKS_NAME: checks_payload,
        VERIFICATION_REPORT_NAME: report_payload,
        SUCCESS_NAME: success_payload,
    }
    outputs = primary.policy["outputs"]
    _require(
        tuple(outputs.get("verification_required_files", ())) == VERIFICATION_FILES,
        "verification output closed set differs from policy",
    )
    selected_root = (
        Path(artifact_root).absolute()
        if artifact_root is not None
        else (project_root / str(outputs["artifact_root"])).absolute()
    )
    secure_root = (
        Path(trusted_root).absolute()
        if trusted_root is not None
        else project_root.absolute()
    )
    destination = (
        selected_root
        / str(outputs["verification_namespace"])
        / primary.build_id
        / verification_id
    )
    pointer_path = (
        Path(verified_pointer_path).absolute()
        if verified_pointer_path is not None
        else (project_root / str(outputs["latest_verified_pointer"])).absolute()
    )

    def gate() -> None:
        primary.gate()
        _require(
            _verification_code_hashes(project_root=project_root) == code_hashes,
            "verification code changed",
        )

    with SecureStagingDirectory.create(
        trusted_root=secure_root,
        destination=destination,
        expected_files=VERIFICATION_FILES,
        label="Stage 10 independent allocator verification",
    ) as stage:
        hashes: dict[str, str] = {}
        for name in VERIFICATION_FILES:
            with stage.open_exclusive_binary(name) as stream:
                written = stream.write(payloads[name])
                _require(
                    written == len(payloads[name]), f"short verification write: {name}"
                )
            hashes[name] = _sha256_bytes(payloads[name])
        snapshot = stage.snapshot(expected_hashes=hashes)
        bundle = stage.publish(
            expected_hashes=snapshot.hashes,
            pre_publish_gate=gate,
            post_publish_gate=gate,
        )
    pointer = {
        "schema_version": 1,
        "contract_id": primary.receipt.contract_id,
        "policy_id": primary.receipt.policy_id,
        "rules_id": primary.receipt.rules_id,
        "build_id": primary.build_id,
        "verification_id": verification_id,
        "status": "VERIFIED_PASSED",
        "primary_manifest_sha256": primary.manifest_sha256,
        "verification_report_path": destination.relative_to(secure_root)
        .joinpath(VERIFICATION_REPORT_NAME)
        .as_posix(),
        "verification_report_sha256": _sha256_bytes(report_payload),
        "verification_checks_sha256": _sha256_bytes(checks_payload),
        "authorizations": VERIFIED_AUTHORIZATIONS,
    }
    replace_pointer_last(
        pointer_path,
        canonical_json_bytes(pointer),
        trusted_root=secure_root,
        pre_replace_gate=gate,
        post_replace_gate=gate,
        label="Stage 10 latest verified allocator pointer",
    )
    return verification_id, destination, pointer_path, bundle


def verify_stage10_allocators(
    *,
    policy_path: Path = DEFAULT_POLICY,
    project_root: Path = PROJECT_ROOT,
    primary_pointer_path: Path | None = None,
    artifact_root: Path | None = None,
    verified_pointer_path: Path | None = None,
    trusted_root: Path | None = None,
    focused_test_receipt: RegressionReceipt | None = None,
    full_test_receipt: RegressionReceipt | None = None,
    synthetic_test_mode: bool = False,
    publish: bool = True,
) -> AllocatorVerificationResult:
    """Authenticate, independently replay, regress, and optionally publish."""

    injected = focused_test_receipt is not None or full_test_receipt is not None
    if injected:
        _require(
            synthetic_test_mode, "injected test receipts require synthetic_test_mode"
        )
        _require(
            focused_test_receipt is not None and full_test_receipt is not None,
            "both test receipts are required",
        )
        _require(
            focused_test_receipt.source == "injected_synthetic_test"
            and full_test_receipt.source == "injected_synthetic_test",
            "synthetic receipts must be marked injected",
        )
        if publish:
            _require(
                trusted_root is not None
                and Path(trusted_root).absolute() != project_root.absolute(),
                "injected receipts cannot publish to the real authority root",
            )
    else:
        _require(
            not synthetic_test_mode, "synthetic_test_mode requires injected receipts"
        )

    with authenticate_primary_allocator(
        policy_path=policy_path,
        project_root=project_root,
        primary_pointer_path=primary_pointer_path,
        trusted_root=trusted_root,
    ) as primary:
        replay = build_synthetic_allocator_evidence()
        published_evidence = primary.manifest["synthetic_evidence"]
        _require(
            canonical_json_bytes(replay) == canonical_json_bytes(published_evidence),
            "independent synthetic evidence replay differs",
        )
        manual = independent_allocator_math()
        _require(
            PRODUCTION_TRAINING_RUNTIME_CERTIFIED is False,
            "decision production gate unexpectedly opened",
        )
        _require(
            published_evidence["scope"][
                "decision_focused_empirical_production_training_ready"
            ]
            is False,
            "primary overclaims decision-focused production readiness",
        )
        if not injected:
            focused_test_receipt, full_test_receipt = run_regression_tests(
                project_root=project_root
            )
        assert focused_test_receipt is not None and full_test_receipt is not None
        _require(
            focused_test_receipt.label == "focused_stage10",
            "focused receipt label differs",
        )
        _require(
            full_test_receipt.label == "full_repository", "full receipt label differs"
        )
        _require(
            focused_test_receipt.passed, "focused Stage 10 regression did not pass"
        )
        _require(full_test_receipt.passed, "full repository regression did not pass")
        checks: dict[str, Any] = {
            "primary_pointer_authenticated": True,
            "primary_closed_bundle_files": len(PRIMARY_FILES),
            "primary_dependency_hashes_authenticated": len(
                primary.manifest["bindings"]["code_hashes"]
            ),
            "primary_configuration_hashes_authenticated": len(
                primary.manifest["bindings"]["configuration_hashes"]
            ),
            "synthetic_evidence_replay_sha256": synthetic_evidence_sha256(replay),
            "independent_manual_check_count": len(manual["checks"]),
            "independent_manual_math_passed": True,
            "focused_stage10_regression_passed": True,
            "full_repository_regression_passed": True,
            "decision_focused_empirical_production_training_ready": False,
            "real_data_payload_rows_read": 0,
            "validation_or_final_rows_read": 0,
            "real_portfolio_weights_trades_or_pnl_created": 0,
            "difference_count": 0,
        }
        verification_id: str | None = None
        verification_path: Path | None = None
        pointer_path: Path | None = None
        bundle: PublishedBundle | None = None
        if publish:
            verification_id, verification_path, pointer_path, bundle = (
                _publish_verification(
                    primary,
                    checks,
                    focused_test_receipt,
                    full_test_receipt,
                    project_root=project_root,
                    artifact_root=artifact_root,
                    verified_pointer_path=verified_pointer_path,
                    trusted_root=trusted_root,
                )
            )
        return AllocatorVerificationResult(
            passed=True,
            build_id=primary.build_id,
            verification_id=verification_id,
            checks=checks,
            verification_path=verification_path,
            verified_pointer_path=pointer_path,
            bundle=bundle,
        )


__all__ = [
    "AllocatorVerificationError",
    "AllocatorVerificationResult",
    "AuthenticatedPrimary",
    "RegressionReceipt",
    "VERIFICATION_CHECKS_NAME",
    "VERIFICATION_CODE_PATHS",
    "VERIFICATION_FILES",
    "VERIFICATION_REPORT_NAME",
    "VERIFIED_AUTHORIZATIONS",
    "authenticate_primary_allocator",
    "independent_allocator_math",
    "run_regression_tests",
    "verification_checks_schema",
    "verify_stage10_allocators",
]
