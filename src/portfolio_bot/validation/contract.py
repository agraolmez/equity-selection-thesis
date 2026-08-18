"""Hash-bound, no-outcome contract for roadmap Stage 11 preflight.

This module authenticates repository-resident policy and authority metadata
only.  It has no database-root argument and no generic payload reader.  The
separate access broker may expose date/column-bounded validation predictors,
but never forward returns, portfolio weights, trades, P&L, or final-period
data.  Validation outcomes require a later approval and verified preflight
pointer.
"""

from __future__ import annotations

import hashlib
import hmac
import importlib.metadata
import json
import os
import re
import stat
import sys
import tomllib
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Callable, Mapping

from portfolio_bot.validation.authorization import (
    ApprovalReceipt,
    ApprovalScope,
    AuthorizationError,
    require_implementation_preflight_authorization,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_POLICY = (
    PROJECT_ROOT
    / "configs/phase_05_validation/stage_11_preflight_policy_v1.toml"
)

POLICY_ID = "phase05-stage11-preflight-policy-v1"
CONTRACT_ID = "phase05-stage11-preflight-v1"
RULES_ID = "phase05-stage11-preflight-rules-v1"
STAGE = 11

METHODOLOGY_SOURCE_SHA256 = (
    "77ac9a4a79b0f46188b56f9a9a3d4affccdbfb13feafdc0f5facc79f2bbc6e81"
)
METHODOLOGY_FREEZE_SHA256 = (
    "cbd897684639a2ccfe1ecf266cf458ef836017b91310fed2e266a75c38808d98"
)
PREFLIGHT_RULES_SHA256 = (
    "0f7810424dfb968c1e02f0d74169c3a19ce833366c940a95d89a33ee16a541c9"
)
PREFLIGHT_POLICY_SHA256 = (
    "030ef9cc162b91001d89ae5d4beace423476b259c69f6c89e7a8a5933c6768b8"
)
PREFLIGHT_APPROVAL_FILE_SHA256 = (
    "31bf22eaeabd9c47b9751bfbe4197bbec45bf167185ad677eec4e963bce49fec"
)
PREFLIGHT_APPROVAL_PAYLOAD_SHA256 = (
    "844da5436a2d1ef8fbe0dcc465213d4e29596d9b5e38b1d23ee6d2aa4ee17748"
)
STAGE10_VERIFIED_POINTER_SHA256 = (
    "af78595e3cd6f35535dd9d30ac018c6c053dfb9e798ff8f00779cf0a451d58a2"
)
STAGE10_BUILD_ID = "phase04-allocator-3f10636b529b111e9d78368e"
STAGE10_VERIFICATION_ID = "verify-c2d5b39a731dc397"
STAGE10_PRIMARY_MANIFEST_SHA256 = (
    "18913c20a3adcf1db498038d3b26913fceb62097d59dde2329387ebdafcdbeba"
)
STAGE10_POLICY_SHA256 = (
    "96c74f9a424bdf2b85f87f82c09ff326808e9cf5a85a8099743e848b17361add"
)
STAGE10_RULES_SHA256 = (
    "a5113e0bdd2d62429312c81d78d5f06e55573bddb775881eb90826c24dcfd851"
)

PREFLIGHT_APPROVAL_ID = "stage11-preflight-approval-20260722"
PREFLIGHT_APPROVAL_SCOPE = ApprovalScope.IMPLEMENTATION_NO_OUTCOME_PREFLIGHT

REQUIRED_RUNTIME_VERSIONS: Mapping[str, str] = {
    "numpy": "2.1.3",
    "pandas": "2.2.3",
    "pyarrow": "19.0.0",
    "torch": "2.7.1",
    "cvxpy": "1.9.2",
    "cvxpylayers": "1.2.0",
    "clarabel": "0.11.1",
    "scs": "3.2.11",
    "diffcp": "1.1.9",
}

REQUIRED_PREFLIGHT_CHECK_NAMES = (
    "methodology_and_stage10_authority_authenticated",
    "preflight_approval_authenticated_and_narrow",
    "validation_and_final_outcome_access_zero",
    "real_weights_trades_and_pnl_zero",
    "exact_59_month_calendar_contract",
    "december_2019_and_2020_plus_rejected",
    "forecast_precedes_outcome",
    "only_completed_labels_enter_refit_or_calibration",
    "development_slice_builder_parity_or_explicit_reconciliation",
    "dynamic_candidate_and_inherited_holding_path_synthetic_pass",
    "required_exit_and_corporate_action_path_synthetic_pass",
    "daily_fixed_share_accounting_independent_parity_pass",
    "transaction_borrow_and_financing_debit_parity_pass",
    "ordinary_projection_and_emergency_rules_pass",
    "decision_focused_empirical_boundary_fail_closed",
    "negative_authorization_and_hash_substitution_tests_pass",
)

PERMITTED_INPUT_ROLES = (
    "repository_metadata",
    "authenticated_stage10_metadata",
    "synthetic_fixture",
    "development_parity_fixture",
    "validation_calendar_metadata",
    "validation_predictor_payload",
)
PROHIBITED_INPUT_ROLES = (
    "validation_outcome_payload",
    "validation_forward_return",
    "real_validation_weight",
    "real_validation_trade",
    "real_validation_pnl_or_performance",
    "final_period_predictor_or_outcome",
    "stage12_or_later_payload",
)

REQUIRED_STAGE10_AUTHORIZATIONS: Mapping[str, bool] = {
    "independent_verification_completed": True,
    "stage_11_validation_technical_gate_passed": True,
    "stage_11_requires_explicit_user_approval": True,
    "stage_11_execution_authorized_now": False,
    "stage_11_execution_started": False,
    "validation_2015_2019_access_authorized_now": False,
    "real_portfolio_weights_trades_or_pnl_authorized_now": False,
    "final_2020_2025_access_authorized": False,
    "stage_12_or_later_authorized": False,
}
REQUIRED_STAGE10_PRIMARY_AUTHORIZATIONS: Mapping[str, bool] = {
    "independent_verification_authorized": True,
    "independent_verification_completed": False,
    "real_pnl_or_performance_authorized": False,
    "real_portfolio_weights_or_trades_authorized": False,
    "stage_11_or_later_authorized": False,
    "stage_11_validation_authorized": False,
    "stage_12_or_later_authorized": False,
    "validation_or_final_test_access_authorized": False,
}
REQUIRED_PREFLIGHT_AUTHORIZATIONS: Mapping[str, bool] = {
    "stage11_implementation_authorized": True,
    "stage11_no_outcome_preflight_authorized": True,
    "development_only_parity_checks_authorized": True,
    "validation_calendar_metadata_authorized": True,
    "guarded_validation_predictor_payload_access_authorized": True,
    "validation_outcome_or_forward_return_access_authorized": False,
    "real_validation_portfolio_weights_or_trades_authorized": False,
    "real_validation_pnl_or_performance_authorized": False,
    "final_2020_2025_access_authorized": False,
    "stage12_or_later_authorized": False,
}
_POLICY_KEYS = frozenset(
    {
        "schema_version",
        "policy_id",
        "contract_id",
        "stage",
        "purpose",
        "scientific_authority",
        "physical_rules",
        "preflight_approval",
        "upstream",
        "scope",
        "runtime",
        "calendar",
        "access_broker",
        "execution_boundary",
        "terminal_accounting",
        "publication",
        "counters_required_zero",
        "counters_required_audited",
    }
)
_RULES_KEYS = frozenset(
    {
        "schema_version",
        "rules_id",
        "scientific_authority",
        "scope",
        "immediate_authority",
        "approval",
        "validation_calendar",
        "locked_validation_choices",
        "walk_forward",
        "portfolio_and_accounting",
        "terminal_accounting",
        "decision_focused_boundary",
        "preflight_required_checks",
        "publication",
    }
)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class Stage11PreflightContractError(RuntimeError):
    """Raised when Stage 11 preflight authority or configuration differs."""


class Stage11PreflightScopeError(Stage11PreflightContractError):
    """Raised when a no-outcome preflight counter crosses its boundary."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Stage11PreflightContractError(message)


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


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    """Hash one regular non-symlinked file through the guarded reader."""

    root = Path(os.path.abspath(os.fspath(path.parent)))
    return _read_project_file(path.name, project_root=root, label="hash input").sha256


def _exact_keys(value: object, expected: frozenset[str], *, label: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"Stage 11 {label} is not a mapping")
    observed = frozenset(str(key) for key in value)
    _require(
        observed == expected,
        f"Stage 11 {label} fields differ; "
        f"missing={sorted(expected - observed)}, extra={sorted(observed - expected)}",
    )
    return value


def _require_sha256(value: object, *, label: str) -> str:
    _require(
        isinstance(value, str) and _SHA256.fullmatch(value) is not None,
        f"Stage 11 {label} is not a lowercase SHA-256",
    )
    return value


def _stat_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
        int(value.st_mode),
        int(value.st_nlink),
        int(value.st_uid),
        int(value.st_gid),
    )


@dataclass(frozen=True, slots=True)
class _AuthenticatedFile:
    path: Path
    payload: bytes
    sha256: str


def _project_path(
    value: str | Path,
    *,
    project_root: Path,
    label: str,
    require_relative: bool,
) -> tuple[Path, Path]:
    root = Path(os.path.abspath(os.fspath(Path(project_root).expanduser())))
    try:
        root_stat = os.lstat(root)
    except OSError as exc:
        raise Stage11PreflightContractError(
            f"Stage 11 project root is absent: {root}"
        ) from exc
    _require(
        stat.S_ISDIR(root_stat.st_mode) and not stat.S_ISLNK(root_stat.st_mode),
        "Stage 11 project root must be a non-symlink directory",
    )

    raw = Path(value).expanduser()
    if require_relative:
        _require(
            not raw.is_absolute() and bool(raw.parts) and ".." not in raw.parts,
            f"Stage 11 {label} must be a safe project-relative path",
        )
    candidate = raw if raw.is_absolute() else root / raw
    path = Path(os.path.abspath(os.fspath(candidate)))
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise Stage11PreflightContractError(
            f"Stage 11 {label} escaped the project root"
        ) from exc
    _require(relative.parts, f"Stage 11 {label} cannot be the project root")

    current = root
    for part in relative.parts:
        current = current / part
        try:
            observed = os.lstat(current)
        except OSError as exc:
            raise Stage11PreflightContractError(
                f"Missing Stage 11 {label}: {current}"
            ) from exc
        _require(
            not stat.S_ISLNK(observed.st_mode),
            f"Symlinked Stage 11 {label} path is prohibited: {current}",
        )
    return root, path


def _read_project_file(
    value: str | Path,
    *,
    project_root: Path,
    label: str,
    expected_sha256: object | None = None,
    require_relative: bool = True,
) -> _AuthenticatedFile:
    """Read one stable regular project file without following symlinks."""

    _, path = _project_path(
        value,
        project_root=project_root,
        label=label,
        require_relative=require_relative,
    )
    expected = (
        None
        if expected_sha256 is None
        else _require_sha256(expected_sha256, label=f"{label} expected hash")
    )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise Stage11PreflightContractError(
            f"Cannot open Stage 11 {label}: {path}"
        ) from exc
    try:
        before = os.fstat(descriptor)
        _require(
            stat.S_ISREG(before.st_mode),
            f"Stage 11 {label} is not a regular file",
        )
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 8 << 20)
            if not block:
                break
            chunks.append(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    _require(
        _stat_identity(before) == _stat_identity(after),
        f"Stage 11 {label} changed while being read",
    )
    try:
        path_after = os.lstat(path)
    except OSError as exc:
        raise Stage11PreflightContractError(
            f"Stage 11 {label} disappeared after read"
        ) from exc
    _require(
        _stat_identity(after) == _stat_identity(path_after),
        f"Stage 11 {label} path identity changed during authentication",
    )
    payload = b"".join(chunks)
    digest = _sha256_bytes(payload)
    if expected is not None:
        _require(
            hmac.compare_digest(digest, expected),
            f"Stage 11 {label} SHA-256 differs",
        )
    return _AuthenticatedFile(path=path, payload=payload, sha256=digest)


def _json(record: _AuthenticatedFile, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(record.payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Stage11PreflightContractError(
            f"Stage 11 {label} is not valid UTF-8 JSON"
        ) from exc
    _require(isinstance(value, dict), f"Stage 11 {label} is not a JSON object")
    return value


def _toml(record: _AuthenticatedFile, *, label: str) -> dict[str, Any]:
    try:
        value = tomllib.loads(record.payload.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise Stage11PreflightContractError(
            f"Stage 11 {label} is not valid UTF-8 TOML"
        ) from exc
    _require(isinstance(value, dict), f"Stage 11 {label} is not a TOML table")
    return value


def _boolean_map_exact(
    observed: object,
    expected: Mapping[str, bool],
    *,
    label: str,
) -> None:
    _require(isinstance(observed, Mapping), f"Stage 11 {label} is missing")
    _require(
        dict(observed) == dict(expected),
        f"Stage 11 {label} differs",
    )


@dataclass(frozen=True, slots=True)
class Stage11PreflightCounters:
    """Immutable audit counters for the guarded no-outcome preflight."""

    validation_predictor_rows_read: int = 0
    validation_predictor_maximum_signal_date: str | None = None
    validation_predictor_forward_return_or_label_columns_read: int = 0
    validation_outcome_rows_read: int = 0
    final_period_rows_read: int = 0
    real_validation_weights_created: int = 0
    real_validation_trades_created: int = 0
    real_validation_pnl_or_performance_rows_created: int = 0

    def to_dict(self) -> dict[str, int | str | None]:
        return {
            "validation_predictor_rows_read": self.validation_predictor_rows_read,
            "validation_predictor_maximum_signal_date": (
                self.validation_predictor_maximum_signal_date
            ),
            "validation_predictor_forward_return_or_label_columns_read": (
                self.validation_predictor_forward_return_or_label_columns_read
            ),
            "validation_outcome_rows_read": self.validation_outcome_rows_read,
            "final_period_rows_read": self.final_period_rows_read,
            "real_validation_weights_created": self.real_validation_weights_created,
            "real_validation_trades_created": self.real_validation_trades_created,
            "real_validation_pnl_or_performance_rows_created": (
                self.real_validation_pnl_or_performance_rows_created
            ),
        }


BoundaryCounters = Stage11PreflightCounters


def enforce_stage11_preflight_boundary(
    counters: Stage11PreflightCounters = Stage11PreflightCounters(),
) -> None:
    if type(counters) is not Stage11PreflightCounters:
        raise Stage11PreflightScopeError(
            "Stage 11 preflight counters must use the immutable counter type"
        )
    for name in (
        "validation_outcome_rows_read",
        "final_period_rows_read",
        "real_validation_weights_created",
        "real_validation_trades_created",
        "real_validation_pnl_or_performance_rows_created",
        "validation_predictor_forward_return_or_label_columns_read",
    ):
        value = getattr(counters, name)
        if type(value) is not int or value != 0:
            raise Stage11PreflightScopeError(
                f"Stage 11 no-outcome preflight prohibits nonzero {name}"
            )
    predictor_rows = counters.validation_predictor_rows_read
    if type(predictor_rows) is not int or predictor_rows < 0:
        raise Stage11PreflightScopeError(
            "Stage 11 audited validation_predictor_rows_read must be a nonnegative integer"
        )
    maximum_date = counters.validation_predictor_maximum_signal_date
    if predictor_rows == 0:
        if maximum_date is not None:
            raise Stage11PreflightScopeError(
                "Stage 11 zero predictor rows require a null audited maximum signal date"
            )
        return
    if not isinstance(maximum_date, str):
        raise Stage11PreflightScopeError(
            "Stage 11 predictor reads require an audited maximum signal date"
        )
    try:
        parsed_maximum = date.fromisoformat(maximum_date)
    except ValueError as exc:
        raise Stage11PreflightScopeError(
            "Stage 11 audited predictor maximum signal date is invalid"
        ) from exc
    if parsed_maximum > date(2019, 10, 31):
        raise Stage11PreflightScopeError(
            "Stage 11 audited predictor maximum signal date exceeds 2019-10-31"
        )


def _validate_scope(policy: Mapping[str, Any], rules: Mapping[str, Any]) -> None:
    expected_policy_scope = {
        "implementation_permitted": True,
        "no_outcome_preflight_permitted": True,
        "synthetic_fixture_execution_permitted": True,
        "development_only_parity_check_permitted": True,
        "validation_calendar_metadata_permitted": True,
        "validation_predictor_payload_access_permitted": True,
        "validation_predictor_access_requires_guarded_column_and_date_allowlist": True,
        "validation_predictor_may_include_forward_returns_or_labels": False,
        "validation_outcome_or_forward_return_access_permitted": False,
        "real_validation_weights_or_trades_permitted": False,
        "real_validation_pnl_or_performance_permitted": False,
        "validation_model_or_ensemble_selection_permitted": False,
        "final_2020_2025_access_permitted": False,
        "stage12_or_later_execution_permitted": False,
    }
    _require(
        policy.get("scope") == expected_policy_scope,
        "Stage 11 policy scope is not the exact no-outcome preflight scope",
    )
    expected_rule_scope = {
        "stage": 11,
        "purpose": (
            "implement_and_verify_validation_runner_without_opening_"
            "validation_or_final_outcomes"
        ),
        "implementation_and_no_outcome_preflight_permitted": True,
        "synthetic_fixture_execution_permitted": True,
        "development_only_parity_checks_permitted": True,
        "validation_calendar_metadata_permitted": True,
        "validation_predictor_payload_access_permitted": True,
        "validation_predictor_access_requires_guarded_column_and_date_allowlist": True,
        "validation_predictor_access_may_include_forward_returns_or_labels": False,
        "validation_outcome_or_forward_return_access_permitted": False,
        "real_validation_portfolio_weights_or_trades_permitted": False,
        "real_validation_pnl_or_performance_permitted": False,
        "validation_selector_or_ensemble_selection_permitted": False,
        "final_2020_2025_payload_access_permitted": False,
        "stage_12_or_later_execution_permitted": False,
    }
    _require(
        rules.get("scope") == expected_rule_scope,
        "Stage 11 physical-rule scope is not the exact no-outcome scope",
    )


def _validate_roles(policy: Mapping[str, Any]) -> None:
    broker = _exact_keys(
        policy.get("access_broker"),
        frozenset(
            {
                "permitted_roles",
                "prohibited_roles",
                "direct_storage_reader_outside_broker_permitted",
                "read_then_filter_later_period_payload_permitted",
            }
        ),
        label="access-broker policy",
    )
    _require(
        tuple(broker.get("permitted_roles", ())) == PERMITTED_INPUT_ROLES,
        "Stage 11 permitted input roles contain an unknown or missing role",
    )
    _require(
        tuple(broker.get("prohibited_roles", ())) == PROHIBITED_INPUT_ROLES,
        "Stage 11 prohibited input roles contain an unknown or missing role",
    )
    _require(
        set(PERMITTED_INPUT_ROLES).isdisjoint(PROHIBITED_INPUT_ROLES),
        "Stage 11 input-role domains overlap",
    )
    _require(
        broker.get("direct_storage_reader_outside_broker_permitted") is False
        and broker.get("read_then_filter_later_period_payload_permitted") is False,
        "Stage 11 access broker permits an unsafe storage path",
    )


def _validate_calendar(policy: Mapping[str, Any], rules: Mapping[str, Any]) -> None:
    expected_policy = {
        "first_holding_month": "2015-01",
        "last_holding_month": "2019-11",
        "complete_holding_months": 59,
        "first_signal_date": "2014-12-31",
        "first_entry_date": "2015-01-02",
        "first_exit_date": "2015-02-02",
        "last_signal_date": "2019-10-31",
        "last_entry_date": "2019-11-01",
        "last_exit_date": "2019-12-02",
        "embargoed_holding_month": "2019-12",
        "first_prohibited_final_entry": "2020-01-02",
    }
    _require(
        policy.get("calendar") == expected_policy,
        "Stage 11 validation calendar boundary differs",
    )
    rule_calendar = rules.get("validation_calendar")
    _require(isinstance(rule_calendar, Mapping), "Stage 11 rule calendar is missing")
    crosswalk = {
        "holding_month_first": "first_holding_month",
        "holding_month_last": "last_holding_month",
        "complete_holding_month_count": "complete_holding_months",
        "first_signal_date": "first_signal_date",
        "first_entry_date": "first_entry_date",
        "first_exit_date": "first_exit_date",
        "last_signal_date": "last_signal_date",
        "last_entry_date": "last_entry_date",
        "last_exit_date": "last_exit_date",
        "embargoed_holding_month": "embargoed_holding_month",
        "first_final_test_entry_date": "first_prohibited_final_entry",
    }
    for rule_name, policy_name in crosswalk.items():
        _require(
            rule_calendar.get(rule_name) == expected_policy[policy_name],
            f"Stage 11 policy/rule calendar differs for {rule_name}",
        )
    _require(
        rule_calendar.get("embargoed_outcome_exit_date") == "2020-01-02"
        and rule_calendar.get("signal_after_month_end_close") is True
        and rule_calendar.get("execution_at_first_following_trading_day_close")
        is True
        and rule_calendar.get("signal_to_execution_return_credited_to_new_portfolio")
        is False
        and rule_calendar.get("forecast_must_precede_entry_and_exit") is True
        and rule_calendar.get("training_label_must_be_completed_before_refit_signal")
        is True
        and rule_calendar.get("one_holding_period_purge_required") is True,
        "Stage 11 rule calendar timing protections differ",
    )


def _validate_definition(
    policy: Mapping[str, Any],
    rules: Mapping[str, Any],
    approval: ApprovalReceipt,
) -> None:
    _exact_keys(policy, _POLICY_KEYS, label="policy")
    _exact_keys(rules, _RULES_KEYS, label="physical rules")
    _require(
        policy.get("schema_version") == 1
        and policy.get("policy_id") == POLICY_ID
        and policy.get("contract_id") == CONTRACT_ID
        and policy.get("stage") == STAGE
        and policy.get("purpose")
        == "hash_bound_no_outcome_validation_runner_preflight",
        "Stage 11 policy identity differs",
    )
    _require(
        rules.get("schema_version") == 1 and rules.get("rules_id") == RULES_ID,
        "Stage 11 physical-rules identity differs",
    )

    scientific = _exact_keys(
        policy.get("scientific_authority"),
        frozenset(
            {
                "methodology_source",
                "methodology_source_sha256",
                "methodology_freeze",
                "methodology_freeze_sha256",
            }
        ),
        label="scientific-authority policy",
    )
    _require(
        scientific
        == {
            "methodology_source": "outputs/methodology_v1_source.md",
            "methodology_source_sha256": METHODOLOGY_SOURCE_SHA256,
            "methodology_freeze": "outputs/methodology_v1_freeze.json",
            "methodology_freeze_sha256": METHODOLOGY_FREEZE_SHA256,
        },
        "Stage 11 methodology authority differs",
    )
    rule_scientific = rules.get("scientific_authority")
    _require(
        isinstance(rule_scientific, Mapping)
        and rule_scientific.get("methodology_source")
        == scientific["methodology_source"]
        and rule_scientific.get("methodology_source_sha256")
        == scientific["methodology_source_sha256"]
        and rule_scientific.get("methodology_freeze")
        == scientific["methodology_freeze"]
        and rule_scientific.get("methodology_freeze_sha256")
        == scientific["methodology_freeze_sha256"],
        "Stage 11 policy/rule methodology binding differs",
    )

    physical = _exact_keys(
        policy.get("physical_rules"),
        frozenset({"path", "sha256", "required_rules_id"}),
        label="physical-rules binding",
    )
    _require(
        physical.get("path")
        == "configs/phase_05_validation/stage_11_preflight_rules_v1.json"
        and physical.get("sha256") == PREFLIGHT_RULES_SHA256
        and physical.get("required_rules_id") == RULES_ID,
        "Stage 11 physical-rules binding differs",
    )
    approval_binding = _exact_keys(
        policy.get("preflight_approval"),
        frozenset(
            {
                "path",
                "required_receipt_id",
                "required_scope",
                "approved",
                "receipt_binds_this_policy_sha256",
                "receipt_hash_is_bound_in_the_preflight_contract_and_manifest",
            }
        ),
        label="preflight-approval binding",
    )
    _require(
        approval_binding.get("path")
        == "configs/phase_05_validation/stage_11_preflight_approval_v1.json"
        and approval_binding.get("required_receipt_id") == PREFLIGHT_APPROVAL_ID
        and approval_binding.get("required_scope") == PREFLIGHT_APPROVAL_SCOPE.value
        and approval_binding.get("approved") is True
        and approval_binding.get("receipt_binds_this_policy_sha256") is True
        and approval_binding.get(
            "receipt_hash_is_bound_in_the_preflight_contract_and_manifest"
        )
        is True,
        "Stage 11 preflight-approval binding differs",
    )

    _validate_scope(policy, rules)
    _validate_roles(policy)
    _validate_calendar(policy, rules)
    _require(
        tuple(rules.get("preflight_required_checks", ()))
        == REQUIRED_PREFLIGHT_CHECK_NAMES,
        "Stage 11 required preflight check set differs",
    )

    execution = policy.get("execution_boundary")
    _require(
        isinstance(execution, Mapping)
        and execution.get("preflight_may_call_validation_entrypoint") is False
        and execution.get("validation_requires_separate_unblind_approval_receipt")
        is True
        and execution.get("validation_requires_verified_preflight_pointer") is True
        and execution.get("final_period_may_be_opened_in_stage11") is False,
        "Stage 11 execution boundary differs",
    )
    publication = policy.get("publication")
    _require(
        isinstance(publication, Mapping)
        and publication.get("pointer_replaced_last") is True
        and publication.get("outcome_or_performance_payload_permitted") is False,
        "Stage 11 publication boundary differs",
    )
    rule_publication = rules.get("publication")
    _require(
        isinstance(rule_publication, Mapping)
        and rule_publication.get("outcome_or_performance_payload_in_preflight_bundle_permitted")
        is False
        and rule_publication.get("validation_execution_entrypoint_must_differ")
        is True,
        "Stage 11 rule publication boundary differs",
    )
    decision = rules.get("decision_focused_boundary")
    _require(
        isinstance(decision, Mapping)
        and decision.get("empirical_training_permitted_in_preflight") is False
        and decision.get("unsupported_rule")
        == "remain_explicitly_inactive_without_approximate_fallback",
        "Stage 11 decision-focused preflight boundary differs",
    )

    counters = policy.get("counters_required_zero")
    _require(
        counters
        == {
            "validation_outcome_rows_read": 0,
            "final_period_rows_read": 0,
            "real_validation_weights_created": 0,
            "real_validation_trades_created": 0,
            "real_validation_pnl_or_performance_rows_created": 0,
        },
        "Stage 11 configured forbidden preflight counters are not exactly zero",
    )
    _require(
        policy.get("counters_required_audited")
        == {
            "validation_predictor_rows_read": True,
            "validation_predictor_maximum_signal_date": "2019-10-31",
            "validation_predictor_forward_return_or_label_columns_read": 0,
        },
        "Stage 11 guarded predictor audit requirements differ",
    )

    _require(
        approval.receipt_id == PREFLIGHT_APPROVAL_ID
        and approval.scope is PREFLIGHT_APPROVAL_SCOPE
        and approval.approved is True,
        "Stage 11 authenticated preflight approval identity differs",
    )
    rule_approval = rules.get("approval")
    _require(
        rule_approval
        == {
            "preflight_receipt": approval_binding.get("path"),
            "required_scope": PREFLIGHT_APPROVAL_SCOPE.value,
            "receipt_must_bind_stage11_policy_sha256": True,
            "receipt_must_bind_stage10_verified_pointer_sha256": True,
            "receipt_payload_sha256_required": True,
            "current_receipt_may_authorize_guarded_validation_predictor_access": True,
            "current_receipt_may_authorize_validation_outcome_access": False,
            "current_receipt_may_authorize_real_weights_trades_or_pnl": False,
            "current_receipt_may_authorize_final_period_access": False,
            "later_validation_unblind_requires_separate_receipt": True,
            "later_final_period_unblind_requires_separate_stage12_freeze_and_receipt": True,
        },
        "Stage 11 rule approval boundary differs",
    )


def _load_definition(
    policy_path: Path,
    *,
    project_root: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    ApprovalReceipt,
    _AuthenticatedFile,
    _AuthenticatedFile,
    _AuthenticatedFile,
]:
    policy_record = _read_project_file(
        policy_path,
        project_root=project_root,
        label="preflight policy",
        expected_sha256=PREFLIGHT_POLICY_SHA256,
        require_relative=False,
    )
    policy = _toml(policy_record, label="preflight policy")
    physical = policy.get("physical_rules")
    _require(isinstance(physical, Mapping), "Stage 11 physical-rules binding is missing")
    rules_record = _read_project_file(
        physical.get("path", ""),
        project_root=project_root,
        label="preflight physical rules",
        expected_sha256=physical.get("sha256"),
    )
    rules = _json(rules_record, label="preflight physical rules")
    approval_binding = policy.get("preflight_approval")
    _require(
        isinstance(approval_binding, Mapping),
        "Stage 11 preflight-approval binding is missing",
    )
    approval_record = _read_project_file(
        approval_binding.get("path", ""),
        project_root=project_root,
        label="preflight approval",
        expected_sha256=PREFLIGHT_APPROVAL_FILE_SHA256,
    )
    approval_value = _json(approval_record, label="preflight approval")
    try:
        approval = require_implementation_preflight_authorization(
            ApprovalReceipt.from_mapping(approval_value),
            expected_stage11_policy_sha256=policy_record.sha256,
        )
    except AuthorizationError as exc:
        raise Stage11PreflightContractError(
            f"Stage 11 preflight approval is invalid: {exc}"
        ) from exc
    _require(
        hmac.compare_digest(
            approval.payload_sha256,
            PREFLIGHT_APPROVAL_PAYLOAD_SHA256,
        ),
        "Stage 11 preflight approval payload differs from the frozen authority",
    )
    _validate_definition(policy, rules, approval)
    return (
        policy,
        rules,
        approval,
        policy_record,
        rules_record,
        approval_record,
    )


def load_policy(
    path: Path = DEFAULT_POLICY,
    *,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    record = _read_project_file(
        path,
        project_root=project_root,
        label="preflight policy",
        require_relative=False,
    )
    return _toml(record, label="preflight policy")


def load_rules(
    policy: Mapping[str, Any],
    *,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    binding = policy.get("physical_rules")
    _require(isinstance(binding, Mapping), "Stage 11 physical-rules binding is missing")
    record = _read_project_file(
        binding.get("path", ""),
        project_root=project_root,
        label="preflight physical rules",
        expected_sha256=binding.get("sha256"),
    )
    return _json(record, label="preflight physical rules")


def validate_contract(
    policy_path: Path = DEFAULT_POLICY,
    *,
    project_root: Path = PROJECT_ROOT,
) -> tuple[dict[str, Any], dict[str, Any], ApprovalReceipt]:
    policy, rules, approval, *_ = _load_definition(
        policy_path, project_root=project_root
    )
    return policy, rules, approval


def inspect_runtime_versions(
    *,
    version_reader: Callable[[str], str] = importlib.metadata.version,
) -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for name in REQUIRED_RUNTIME_VERSIONS:
        try:
            result[name] = version_reader(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def validate_runtime_versions(
    runtime_policy: object,
    *,
    version_reader: Callable[[str], str] = importlib.metadata.version,
) -> dict[str, str]:
    expected_policy = {"minimum_python": "3.11", **REQUIRED_RUNTIME_VERSIONS}
    _require(
        runtime_policy == expected_policy,
        "Stage 11 runtime policy differs from the frozen versions",
    )
    _require(sys.version_info >= (3, 11), "Stage 11 requires Python 3.11 or later")
    observed = inspect_runtime_versions(version_reader=version_reader)
    for name, expected in REQUIRED_RUNTIME_VERSIONS.items():
        _require(
            observed.get(name) == expected,
            f"Stage 11 runtime version differs for {name}: "
            f"expected {expected}, observed {observed.get(name)}",
        )
    return {name: str(observed[name]) for name in REQUIRED_RUNTIME_VERSIONS}


def _validate_methodology(
    policy: Mapping[str, Any],
    *,
    project_root: Path,
) -> None:
    authority = policy["scientific_authority"]
    source = _read_project_file(
        authority["methodology_source"],
        project_root=project_root,
        label="Methodology v1 source",
        expected_sha256=authority["methodology_source_sha256"],
    )
    freeze_record = _read_project_file(
        authority["methodology_freeze"],
        project_root=project_root,
        label="Methodology v1 freeze",
        expected_sha256=authority["methodology_freeze_sha256"],
    )
    freeze = _json(freeze_record, label="Methodology v1 freeze")
    _require(
        freeze.get("methodology_version") == "v1"
        and freeze.get("status") == "frozen_preimplementation"
        and freeze.get("empirical_results_seen") is False,
        "Stage 11 Methodology v1 freeze state differs",
    )
    entries = [
        item
        for item in freeze.get("files", [])
        if isinstance(item, Mapping)
        and item.get("path") == authority["methodology_source"]
    ]
    _require(
        len(entries) == 1
        and entries[0].get("sha256") == source.sha256,
        "Stage 11 Methodology v1 source is not uniquely frozen",
    )


def _validate_upstream_definition(policy: Mapping[str, Any], rules: Mapping[str, Any]) -> None:
    upstream = _exact_keys(
        policy.get("upstream"),
        frozenset(
            {
                "verified_pointer",
                "verified_pointer_sha256",
                "required_build_id",
                "required_verification_id",
                "required_status",
                "primary_manifest",
                "primary_manifest_sha256",
                "stage10_policy",
                "stage10_policy_sha256",
                "stage10_rules",
                "stage10_rules_sha256",
                "unregistered_input_permitted",
                "hash_bypass_permitted",
                "symlinked_path_permitted",
            }
        ),
        label="upstream authority",
    )
    expected = {
        "verified_pointer": "artifacts/phase_04_models/LATEST_ALLOCATOR_VERIFIED.json",
        "verified_pointer_sha256": STAGE10_VERIFIED_POINTER_SHA256,
        "required_build_id": STAGE10_BUILD_ID,
        "required_verification_id": STAGE10_VERIFICATION_ID,
        "required_status": "VERIFIED_PASSED",
        "primary_manifest": (
            "artifacts/phase_04_models/allocator_builds/"
            f"{STAGE10_BUILD_ID}/manifest.json"
        ),
        "primary_manifest_sha256": STAGE10_PRIMARY_MANIFEST_SHA256,
        "stage10_policy": "configs/phase_04_models/allocation_policy_v1.toml",
        "stage10_policy_sha256": STAGE10_POLICY_SHA256,
        "stage10_rules": "configs/phase_04_models/allocation_build_rules_v1.json",
        "stage10_rules_sha256": STAGE10_RULES_SHA256,
        "unregistered_input_permitted": False,
        "hash_bypass_permitted": False,
        "symlinked_path_permitted": False,
    }
    _require(dict(upstream) == expected, "Stage 11 upstream authority differs")
    rule_authority = rules.get("immediate_authority")
    _require(isinstance(rule_authority, Mapping), "Stage 11 rule authority is missing")
    for name in (
        "verified_pointer",
        "verified_pointer_sha256",
        "required_status",
        "required_build_id",
        "required_verification_id",
        "primary_manifest",
        "primary_manifest_sha256",
        "stage10_policy",
        "stage10_policy_sha256",
        "stage10_rules",
        "stage10_rules_sha256",
    ):
        _require(
            rule_authority.get(name) == expected[name],
            f"Stage 11 policy/rule upstream binding differs for {name}",
        )
    _boolean_map_exact(
        rule_authority.get("required_pointer_authorizations"),
        REQUIRED_STAGE10_AUTHORIZATIONS,
        label="required Stage 10 pointer authorizations",
    )
    _require(
        rule_authority.get("unregistered_input_permitted") is False
        and rule_authority.get("hash_bypass_permitted") is False
        and rule_authority.get("symlinked_authority_path_permitted") is False,
        "Stage 11 rule upstream safety boundary differs",
    )


def _validate_stage10_authority(
    policy: Mapping[str, Any],
    rules: Mapping[str, Any],
    *,
    project_root: Path,
) -> None:
    _validate_upstream_definition(policy, rules)
    upstream = policy["upstream"]
    pointer_record = _read_project_file(
        upstream["verified_pointer"],
        project_root=project_root,
        label="Stage 10 verified pointer",
        expected_sha256=upstream["verified_pointer_sha256"],
    )
    pointer = _json(pointer_record, label="Stage 10 verified pointer")
    _require(
        pointer.get("schema_version") == 1
        and pointer.get("status") == "VERIFIED_PASSED"
        and pointer.get("build_id") == STAGE10_BUILD_ID
        and pointer.get("verification_id") == STAGE10_VERIFICATION_ID
        and pointer.get("contract_id") == "phase04-allocation-v1"
        and pointer.get("policy_id") == "phase04-allocation-policy-v1"
        and pointer.get("rules_id") == "phase04-allocation-build-rules-v1"
        and pointer.get("primary_manifest_sha256")
        == STAGE10_PRIMARY_MANIFEST_SHA256,
        "Stage 11 Stage 10 verified-pointer identity differs",
    )
    _boolean_map_exact(
        pointer.get("authorizations"),
        REQUIRED_STAGE10_AUTHORIZATIONS,
        label="Stage 10 verified-pointer authorizations",
    )

    manifest_record = _read_project_file(
        upstream["primary_manifest"],
        project_root=project_root,
        label="Stage 10 primary manifest",
        expected_sha256=upstream["primary_manifest_sha256"],
    )
    manifest = _json(manifest_record, label="Stage 10 primary manifest")
    _require(
        manifest.get("schema_version") == 1
        and manifest.get("status")
        == "PRIMARY_PASSED_PENDING_INDEPENDENT_VERIFICATION"
        and manifest.get("build_id") == STAGE10_BUILD_ID
        and manifest.get("contract_id") == "phase04-allocation-v1"
        and manifest.get("policy_id") == "phase04-allocation-policy-v1"
        and manifest.get("rules_id") == "phase04-allocation-build-rules-v1",
        "Stage 11 Stage 10 primary-manifest identity differs",
    )
    _boolean_map_exact(
        manifest.get("authorizations"),
        REQUIRED_STAGE10_PRIMARY_AUTHORIZATIONS,
        label="Stage 10 primary-manifest authorizations",
    )
    hard_gates = manifest.get("hard_gates")
    summary = manifest.get("summary")
    _require(
        isinstance(hard_gates, Mapping)
        and hard_gates.get("real_data_payload_rows_read") == 0
        and hard_gates.get("validation_or_final_rows_read") == 0
        and hard_gates.get("real_portfolio_weights_or_trades_created") == 0
        and hard_gates.get("real_pnl_or_performance_rows_created") == 0,
        "Stage 11 Stage 10 empirical hard-gate counters differ",
    )
    _require(
        isinstance(summary, Mapping)
        and summary.get("empirical_security_rows_read") == 0
        and summary.get("empirical_weights_trades_or_pnl_rows_created") == 0
        and summary.get("stage_11_authorized") is False
        and summary.get("decision_focused_empirical_production_training_ready")
        is False,
        "Stage 11 Stage 10 summary boundary differs",
    )

    stage10_policy_record = _read_project_file(
        upstream["stage10_policy"],
        project_root=project_root,
        label="Stage 10 policy",
        expected_sha256=upstream["stage10_policy_sha256"],
    )
    stage10_policy = _toml(stage10_policy_record, label="Stage 10 policy")
    _require(
        stage10_policy.get("schema_version") == 1
        and stage10_policy.get("stage") == 10
        and stage10_policy.get("policy_id") == "phase04-allocation-policy-v1"
        and stage10_policy.get("contract_id") == "phase04-allocation-v1"
        and stage10_policy.get("status")
        == "frozen_definition_only_before_allocator_implementation",
        "Stage 11 Stage 10 policy identity differs",
    )
    _require(
        stage10_policy.get("methodology", {}).get("source_sha256")
        == METHODOLOGY_SOURCE_SHA256
        and stage10_policy.get("methodology", {}).get("freeze_receipt_sha256")
        == METHODOLOGY_FREEZE_SHA256
        and stage10_policy.get("physical_rules", {}).get("sha256")
        == STAGE10_RULES_SHA256,
        "Stage 11 Stage 10 policy bindings differ",
    )

    stage10_rules_record = _read_project_file(
        upstream["stage10_rules"],
        project_root=project_root,
        label="Stage 10 rules",
        expected_sha256=upstream["stage10_rules_sha256"],
    )
    stage10_rules = _json(stage10_rules_record, label="Stage 10 rules")
    _require(
        stage10_rules.get("schema_version") == 1
        and stage10_rules.get("rules_id") == "phase04-allocation-build-rules-v1"
        and stage10_rules.get("scientific_authority", {}).get(
            "methodology_source_sha256"
        )
        == METHODOLOGY_SOURCE_SHA256,
        "Stage 11 Stage 10 rules identity differs",
    )

    bindings = manifest.get("bindings")
    configuration_hashes = (
        bindings.get("configuration_hashes") if isinstance(bindings, Mapping) else None
    )
    _require(
        isinstance(configuration_hashes, Mapping)
        and configuration_hashes.get(upstream["stage10_policy"])
        == STAGE10_POLICY_SHA256
        and configuration_hashes.get(upstream["stage10_rules"])
        == STAGE10_RULES_SHA256
        and configuration_hashes.get("outputs/methodology_v1_source.md")
        == METHODOLOGY_SOURCE_SHA256
        and configuration_hashes.get("outputs/methodology_v1_freeze.json")
        == METHODOLOGY_FREEZE_SHA256,
        "Stage 11 Stage 10 manifest configuration bindings differ",
    )
    code_hashes = bindings.get("code_hashes") if isinstance(bindings, Mapping) else None
    _require(
        isinstance(code_hashes, Mapping) and bool(code_hashes),
        "Stage 11 Stage 10 manifest code bindings are missing",
    )
    for relative_path, expected_sha256 in sorted(code_hashes.items()):
        _require(
            isinstance(relative_path, str)
            and isinstance(expected_sha256, str)
            and _SHA256.fullmatch(expected_sha256) is not None,
            "Stage 11 Stage 10 manifest code binding is malformed",
        )
        live_record = _read_project_file(
            relative_path,
            project_root=project_root,
            label=f"live Stage 10 implementation {relative_path}",
            expected_sha256=expected_sha256,
        )
        _require(
            live_record.sha256 == expected_sha256,
            f"Stage 11 live Stage 10 implementation differs: {relative_path}",
        )


@dataclass(frozen=True, slots=True)
class Stage11PreflightContractReceipt:
    """Immutable authorization receipt for implementation/preflight only."""

    schema_version: int
    status: str
    stage: int
    policy_id: str
    contract_id: str
    rules_id: str
    approval_id: str
    approval_scope: str
    policy_sha256: str
    rules_sha256: str
    approval_file_sha256: str
    approval_payload_sha256: str
    methodology_source_sha256: str
    methodology_freeze_sha256: str
    stage10_verified_pointer_sha256: str
    stage10_build_id: str
    stage10_verification_id: str
    stage10_primary_manifest_sha256: str
    stage10_policy_sha256: str
    stage10_rules_sha256: str
    first_holding_month: str
    last_holding_month: str
    complete_holding_months: int
    first_signal_date: str
    last_exit_date: str
    runtime_versions: tuple[tuple[str, str], ...]
    permitted_input_roles: tuple[str, ...]
    prohibited_input_roles: tuple[str, ...]
    preflight_authorizations: tuple[tuple[str, bool], ...]
    counters: Stage11PreflightCounters

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "stage": self.stage,
            "policy_id": self.policy_id,
            "contract_id": self.contract_id,
            "rules_id": self.rules_id,
            "approval_id": self.approval_id,
            "approval_scope": self.approval_scope,
            "policy_sha256": self.policy_sha256,
            "rules_sha256": self.rules_sha256,
            "approval_file_sha256": self.approval_file_sha256,
            "approval_payload_sha256": self.approval_payload_sha256,
            "methodology_source_sha256": self.methodology_source_sha256,
            "methodology_freeze_sha256": self.methodology_freeze_sha256,
            "stage10_verified_pointer_sha256": self.stage10_verified_pointer_sha256,
            "stage10_build_id": self.stage10_build_id,
            "stage10_verification_id": self.stage10_verification_id,
            "stage10_primary_manifest_sha256": self.stage10_primary_manifest_sha256,
            "stage10_policy_sha256": self.stage10_policy_sha256,
            "stage10_rules_sha256": self.stage10_rules_sha256,
            "first_holding_month": self.first_holding_month,
            "last_holding_month": self.last_holding_month,
            "complete_holding_months": self.complete_holding_months,
            "first_signal_date": self.first_signal_date,
            "last_exit_date": self.last_exit_date,
            "runtime_versions": dict(self.runtime_versions),
            "permitted_input_roles": list(self.permitted_input_roles),
            "prohibited_input_roles": list(self.prohibited_input_roles),
            "preflight_authorizations": dict(self.preflight_authorizations),
            "counters": self.counters.to_dict(),
        }

    @property
    def sha256(self) -> str:
        return _sha256_bytes(canonical_json_bytes(self.to_dict()))


def authenticate_stage11_preflight_contract(
    policy_path: Path = DEFAULT_POLICY,
    *,
    project_root: Path = PROJECT_ROOT,
    version_reader: Callable[[str], str] = importlib.metadata.version,
    counters: Stage11PreflightCounters = Stage11PreflightCounters(),
) -> Stage11PreflightContractReceipt:
    """Authenticate the complete repository-only Stage 11 preflight graph."""

    enforce_stage11_preflight_boundary(counters)
    (
        policy,
        rules,
        approval,
        policy_record,
        rules_record,
        approval_record,
    ) = _load_definition(policy_path, project_root=project_root)
    _validate_methodology(policy, project_root=project_root)
    _validate_stage10_authority(policy, rules, project_root=project_root)
    runtime = validate_runtime_versions(
        policy.get("runtime"), version_reader=version_reader
    )
    calendar = policy["calendar"]
    return Stage11PreflightContractReceipt(
        schema_version=1,
        status="AUTHENTICATED_NO_OUTCOME_PREFLIGHT_ONLY",
        stage=STAGE,
        policy_id=POLICY_ID,
        contract_id=CONTRACT_ID,
        rules_id=RULES_ID,
        approval_id=approval.receipt_id,
        approval_scope=approval.scope.value,
        policy_sha256=policy_record.sha256,
        rules_sha256=rules_record.sha256,
        approval_file_sha256=approval_record.sha256,
        approval_payload_sha256=approval.payload_sha256,
        methodology_source_sha256=METHODOLOGY_SOURCE_SHA256,
        methodology_freeze_sha256=METHODOLOGY_FREEZE_SHA256,
        stage10_verified_pointer_sha256=STAGE10_VERIFIED_POINTER_SHA256,
        stage10_build_id=STAGE10_BUILD_ID,
        stage10_verification_id=STAGE10_VERIFICATION_ID,
        stage10_primary_manifest_sha256=STAGE10_PRIMARY_MANIFEST_SHA256,
        stage10_policy_sha256=STAGE10_POLICY_SHA256,
        stage10_rules_sha256=STAGE10_RULES_SHA256,
        first_holding_month=str(calendar["first_holding_month"]),
        last_holding_month=str(calendar["last_holding_month"]),
        complete_holding_months=int(calendar["complete_holding_months"]),
        first_signal_date=str(calendar["first_signal_date"]),
        last_exit_date=str(calendar["last_exit_date"]),
        runtime_versions=tuple(runtime.items()),
        permitted_input_roles=PERMITTED_INPUT_ROLES,
        prohibited_input_roles=PROHIBITED_INPUT_ROLES,
        preflight_authorizations=tuple(REQUIRED_PREFLIGHT_AUTHORIZATIONS.items()),
        counters=counters,
    )


build_contract_receipt = authenticate_stage11_preflight_contract


__all__ = [
    "BoundaryCounters",
    "CONTRACT_ID",
    "DEFAULT_POLICY",
    "PERMITTED_INPUT_ROLES",
    "POLICY_ID",
    "PREFLIGHT_APPROVAL_FILE_SHA256",
    "PREFLIGHT_APPROVAL_ID",
    "PREFLIGHT_APPROVAL_PAYLOAD_SHA256",
    "PREFLIGHT_APPROVAL_SCOPE",
    "PREFLIGHT_POLICY_SHA256",
    "PROHIBITED_INPUT_ROLES",
    "PROJECT_ROOT",
    "REQUIRED_RUNTIME_VERSIONS",
    "REQUIRED_PREFLIGHT_CHECK_NAMES",
    "RULES_ID",
    "STAGE",
    "STAGE10_BUILD_ID",
    "STAGE10_VERIFICATION_ID",
    "STAGE10_VERIFIED_POINTER_SHA256",
    "Stage11PreflightContractError",
    "Stage11PreflightContractReceipt",
    "Stage11PreflightCounters",
    "Stage11PreflightScopeError",
    "authenticate_stage11_preflight_contract",
    "build_contract_receipt",
    "canonical_json_bytes",
    "enforce_stage11_preflight_boundary",
    "inspect_runtime_versions",
    "load_policy",
    "load_rules",
    "sha256_file",
    "validate_contract",
    "validate_runtime_versions",
]
