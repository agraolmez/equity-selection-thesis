"""Fail-closed Stage 13 orchestration and segment-isolated memoization.

The public production entry accepts no caller dates, paths, readers, or
callbacks.  It authenticates the fixed Stage 12 verification, a separate
fixed Stage 13 approval, and the fixed verified registration pointer before
delegating to the production state machine.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from portfolio_bot.final_test.authorization import Stage12VerifiedBinding
from portfolio_bot.final_test.calendar import (
    CORE_COMPLETE_HOLDING_MONTHS,
    PARTIAL_DECEMBER_HOLDING_MONTH,
    PRIMARY_FULL_HOLDING_MONTHS,
    FinalPeriodSegment,
)
from portfolio_bot.final_test.checkpoint import (
    CheckpointPlan,
    LoadedCheckpoint,
    MonthlyCheckpointRecord,
    MonthlyCheckpointStore,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
STAGE13_CHECKPOINT_ROOT = (
    PROJECT_ROOT / "artifacts/phase_06_final_test/stage_13_checkpoints"
)
STAGE13_REGISTRATION_POINTER = (
    PROJECT_ROOT
    / "artifacts/phase_06_final_test/LATEST_STAGE_13_REGISTRATION_VERIFIED.json"
)
STAGE13_APPROVAL_RECEIPT = (
    PROJECT_ROOT / "artifacts/phase_06_final_test/STAGE_13_APPROVAL.json"
)
ECONOMIC_RUNTIME_STATUS = "CERTIFIED_FINAL_ECONOMIC_KERNEL_CONTRACT_AVAILABLE"
REGISTRATION_RUNTIME_STATUS = "VERIFIED_SCHEMA_BOUND_REGISTRATION_LOADER_AVAILABLE"
STAGE13_STATE_FORMAT = "application/vnd.portfolio-bot.stage13-month+json"

_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class Stage13ExecutionError(RuntimeError):
    """Raised before unblind when the frozen Stage 13 contract is incomplete."""


class Stage13Segment(str, Enum):
    PRIMARY = "primary_full"
    CORE = "core_complete"
    PARTIAL = "partial_december_descriptive"


@dataclass(frozen=True, slots=True)
class FrozenSegmentPlan:
    segment: Stage13Segment
    chain_id: str
    holding_months: tuple[str, ...]
    registration_names: tuple[str, ...]

    def __post_init__(self) -> None:
        expected = {
            Stage13Segment.PRIMARY: PRIMARY_FULL_HOLDING_MONTHS,
            Stage13Segment.CORE: CORE_COMPLETE_HOLDING_MONTHS,
            Stage13Segment.PARTIAL: (PARTIAL_DECEMBER_HOLDING_MONTH,),
        }[self.segment]
        if self.holding_months != expected:
            raise Stage13ExecutionError("segment month domain differs from freeze")
        if not self.registration_names or len(self.registration_names) != len(
            set(self.registration_names)
        ):
            raise Stage13ExecutionError("segment registrations are not exact")


PRIMARY_REGISTRATIONS = (
    "final_calendar",
    "primary_predictors",
    "primary_risk_cost",
    "primary_cost_inputs",
    "primary_factor_covariance",
    "primary_risk_cost_month_summary",
    "primary_market_inputs",
    "primary_market_accounting",
    "primary_factors",
    "primary_benchmarks",
    "primary_completed_labels",
    "primary_outcomes",
)
CORE_REGISTRATIONS = (
    "final_calendar",
    "core_predictors",
    "core_risk_cost",
    "core_cost_inputs",
    "core_factor_covariance",
    "core_risk_cost_month_summary",
    "core_market_inputs",
    "core_market_accounting",
    "core_factors",
    "core_benchmarks",
    "core_completed_labels",
    "core_outcomes",
)
PARTIAL_REGISTRATIONS = (
    "final_calendar",
    "partial_predictors",
    "partial_risk_cost",
    "partial_cost_inputs",
    "partial_factor_covariance",
    "partial_risk_cost_month_summary",
    "partial_market_inputs",
    "partial_market_accounting",
    "partial_benchmarks",
    "partial_descriptive_outcome",
)

FROZEN_SEGMENT_PLANS = (
    FrozenSegmentPlan(
        segment=Stage13Segment.PRIMARY,
        chain_id="stage13-primary-full-v1",
        holding_months=PRIMARY_FULL_HOLDING_MONTHS,
        registration_names=PRIMARY_REGISTRATIONS,
    ),
    FrozenSegmentPlan(
        segment=Stage13Segment.CORE,
        chain_id="stage13-core-complete-v1",
        holding_months=CORE_COMPLETE_HOLDING_MONTHS,
        registration_names=CORE_REGISTRATIONS,
    ),
    FrozenSegmentPlan(
        segment=Stage13Segment.PARTIAL,
        chain_id="stage13-partial-december-v1",
        holding_months=(PARTIAL_DECEMBER_HOLDING_MONTH,),
        registration_names=PARTIAL_REGISTRATIONS,
    ),
)
FROZEN_PLAN_BY_SEGMENT = MappingProxyType(
    {plan.segment: plan for plan in FROZEN_SEGMENT_PLANS}
)


def _load_exact_registry_columns() -> Mapping[str, tuple[str, ...]]:
    path = (
        PROJECT_ROOT
        / "configs/phase_06_final_test/stage_13_input_registry_v1.json"
    )
    try:
        value = json.loads(path.read_bytes())
        templates = value["templates"]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise Stage13ExecutionError("frozen final-input registry is invalid") from exc
    if not isinstance(templates, list):
        raise Stage13ExecutionError("frozen final-input templates are missing")
    selected = {
        name
        for plan in FROZEN_SEGMENT_PLANS
        for name in plan.registration_names
    }
    result: dict[str, tuple[str, ...]] = {}
    for item in templates:
        if not isinstance(item, Mapping):
            raise Stage13ExecutionError("frozen final-input template is invalid")
        name = str(item.get("registration_name"))
        if name not in selected:
            continue
        columns = item.get("required_columns")
        if (
            not isinstance(columns, list)
            or not columns
            or not all(isinstance(column, str) and column for column in columns)
        ):
            raise Stage13ExecutionError("frozen exact column list is invalid")
        result[name] = tuple(columns)
    if set(result) != selected:
        raise Stage13ExecutionError("frozen registration set is incomplete")
    return MappingProxyType(result)


FROZEN_EXACT_COLUMNS = _load_exact_registry_columns()


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_hash(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise Stage13ExecutionError(f"{field} must be an exact SHA-256")
    return value


def frozen_execution_config_sha256() -> str:
    """Bind the exact Stage 12 configs and this orchestration implementation."""

    paths = (
        PROJECT_ROOT
        / "configs/phase_06_final_test/stage_12_final_system_v1.json",
        PROJECT_ROOT
        / "configs/phase_06_final_test/stage_13_input_registry_v1.json",
        PROJECT_ROOT / "src/portfolio_bot/final_test/access.py",
        PROJECT_ROOT / "src/portfolio_bot/final_test/authorization.py",
        PROJECT_ROOT / "src/portfolio_bot/final_test/benchmarks.py",
        PROJECT_ROOT / "src/portfolio_bot/final_test/calendar.py",
        PROJECT_ROOT / "src/portfolio_bot/final_test/checkpoint.py",
        PROJECT_ROOT / "src/portfolio_bot/final_test/economic_kernel.py",
        PROJECT_ROOT / "src/portfolio_bot/final_test/inference.py",
        PROJECT_ROOT / "src/portfolio_bot/final_test/input_materialization.py",
        PROJECT_ROOT / "src/portfolio_bot/final_test/final_input_construction.py",
        PROJECT_ROOT / "src/portfolio_bot/final_test/orchestration.py",
        PROJECT_ROOT / "src/portfolio_bot/final_test/production_results.py",
        PROJECT_ROOT / "src/portfolio_bot/final_test/production_runtime.py",
        PROJECT_ROOT / "src/portfolio_bot/final_test/publication.py",
        PROJECT_ROOT / "src/portfolio_bot/final_test/results.py",
        PROJECT_ROOT / "src/portfolio_bot/numerical_materiality.py",
        PROJECT_ROOT / "src/portfolio_bot/allocation/rank.py",
        PROJECT_ROOT / "src/portfolio_bot/allocation/solver.py",
        PROJECT_ROOT / "src/portfolio_bot/selectors/elastic_net.py",
        PROJECT_ROOT / "src/portfolio_bot/selectors/neural_model.py",
        PROJECT_ROOT / "src/portfolio_bot/selectors/xgboost_model.py",
        PROJECT_ROOT / "src/portfolio_bot/validation/accounting.py",
        PROJECT_ROOT / "src/portfolio_bot/validation/calibration.py",
        PROJECT_ROOT / "src/portfolio_bot/validation/daily_engine.py",
        PROJECT_ROOT
        / "src/portfolio_bot/validation/empirical_selector_backend.py",
        PROJECT_ROOT / "src/portfolio_bot/validation/problem_builder.py",
        PROJECT_ROOT / "src/portfolio_bot/validation/session_authority.py",
        PROJECT_ROOT
        / "src/portfolio_bot/validation/selector_native_runtime.py",
        PROJECT_ROOT / "src/portfolio_bot/validation/selector_runtime.py",
    )
    members: dict[str, str] = {}
    for path in paths:
        try:
            relative = path.relative_to(PROJECT_ROOT).as_posix()
            payload = path.read_bytes()
        except (OSError, ValueError) as exc:
            raise Stage13ExecutionError(
                "frozen execution configuration is unavailable"
            ) from exc
        members[relative] = _sha256_bytes(payload)
    return _sha256_bytes(
        _canonical_json_bytes(
            {
                "schema_version": 1,
                "runtime_status": ECONOMIC_RUNTIME_STATUS,
                "registration_status": REGISTRATION_RUNTIME_STATUS,
                "segment_plans": [
                    {
                        "segment": plan.segment.value,
                        "chain_id": plan.chain_id,
                        "holding_months": list(plan.holding_months),
                        "registration_names": list(plan.registration_names),
                    }
                    for plan in FROZEN_SEGMENT_PLANS
                ],
                "members": members,
            }
        )
    )


@dataclass(frozen=True, slots=True)
class FrozenRegistrationManifestEntry:
    """Exact materialized table identity required by the future fixed kernel."""

    registration_name: str
    segment: Stage13Segment
    relative_path: str
    file_sha256: str
    size_bytes: int
    exact_columns: tuple[str, ...]
    arrow_schema_sha256: str
    holding_months: tuple[str, ...]

    def __post_init__(self) -> None:
        plan = FROZEN_PLAN_BY_SEGMENT[self.segment]
        if self.registration_name not in plan.registration_names:
            raise Stage13ExecutionError("registration is outside its frozen segment")
        selected = Path(self.relative_path)
        if (
            selected.is_absolute()
            or ".." in selected.parts
            or not selected.parts
            or selected.parts[0] not in {"artifacts", "final_inputs"}
        ):
            raise Stage13ExecutionError("registration path is not bounded")
        _require_hash(self.file_sha256, field="file_sha256")
        _require_hash(self.arrow_schema_sha256, field="arrow_schema_sha256")
        if type(self.size_bytes) is not int or self.size_bytes <= 0:
            raise Stage13ExecutionError("registration size must be positive")
        if (
            not self.exact_columns
            or len(self.exact_columns) != len(set(self.exact_columns))
            or self.exact_columns
            != FROZEN_EXACT_COLUMNS[self.registration_name]
            or self.holding_months != plan.holding_months
        ):
            raise Stage13ExecutionError(
                "registration columns/months differ from the frozen template"
            )


@dataclass(frozen=True, slots=True)
class FrozenMonthRequest:
    plan: FrozenSegmentPlan
    holding_month: str
    stage12_binding_sha256: str
    execution_config_sha256: str
    previous_state_sha256: str | None


@dataclass(frozen=True, slots=True)
class FrozenMonthOutput:
    input_sha256_by_registration: Mapping[str, str]
    access_receipt_sha256: str
    output_sha256: str
    runtime_id: str
    synthetic: bool

    def validate_for(self, request: FrozenMonthRequest) -> None:
        if set(self.input_sha256_by_registration) != set(
            request.plan.registration_names
        ):
            raise Stage13ExecutionError("month input receipt set differs")
        for digest in self.input_sha256_by_registration.values():
            _require_hash(digest, field="input_sha256")
        _require_hash(self.access_receipt_sha256, field="access_receipt_sha256")
        _require_hash(self.output_sha256, field="output_sha256")
        if not isinstance(self.runtime_id, str) or not self.runtime_id:
            raise Stage13ExecutionError("runtime identity is missing")


@dataclass(frozen=True, slots=True)
class SegmentExecutionResult:
    plan: FrozenSegmentPlan
    checkpoint_plan_sha256: str
    head: MonthlyCheckpointRecord
    chain_records: tuple[MonthlyCheckpointRecord, ...]
    executed_months: tuple[str, ...]
    resumed_month_count: int
    synthetic: bool

    def __post_init__(self) -> None:
        if (
            len(self.chain_records) != len(self.plan.holding_months)
            or self.chain_records[-1] != self.head
            or tuple(item.holding_month for item in self.chain_records)
            != self.plan.holding_months
            or tuple(item.ordinal for item in self.chain_records)
            != tuple(range(1, len(self.chain_records) + 1))
        ):
            raise Stage13ExecutionError("published checkpoint chain is incomplete")
        for prior, current in zip(self.chain_records, self.chain_records[1:]):
            if current.previous_checkpoint_sha256 != prior.checkpoint_sha256:
                raise Stage13ExecutionError("published checkpoint chain is spliced")


@dataclass(frozen=True, slots=True)
class Stage13RunResult:
    stage12_binding: Stage12VerifiedBinding
    execution_config_sha256: str
    segments: tuple[SegmentExecutionResult, ...]
    synthetic: bool

    def __post_init__(self) -> None:
        if tuple(item.plan.segment for item in self.segments) != tuple(
            plan.segment for plan in FROZEN_SEGMENT_PLANS
        ):
            raise Stage13ExecutionError("Stage 13 segment result set differs")
        if any(item.synthetic is not self.synthetic for item in self.segments):
            raise Stage13ExecutionError("mixed synthetic/empirical results prohibited")


class _SyntheticFrozenRuntime:
    """Private deterministic runtime used only to test crash/resume mechanics."""

    runtime_id = "stage13-synthetic-checkpoint-probe-v1"

    def __init__(
        self,
        *,
        crash_segment: Stage13Segment | None = None,
        crash_month: str | None = None,
    ) -> None:
        self.crash_segment = crash_segment
        self.crash_month = crash_month
        self.executed: list[tuple[Stage13Segment, str]] = []
        self._crashed = False

    def execute(self, request: FrozenMonthRequest) -> FrozenMonthOutput:
        self.executed.append((request.plan.segment, request.holding_month))
        if (
            not self._crashed
            and request.plan.segment is self.crash_segment
            and request.holding_month == self.crash_month
        ):
            self._crashed = True
            raise Stage13ExecutionError("synthetic crash before checkpoint commit")
        inputs = {
            name: _sha256_bytes(
                _canonical_json_bytes(
                    {
                        "synthetic": True,
                        "registration_name": name,
                        "holding_month": request.holding_month,
                        "segment": request.plan.segment.value,
                    }
                )
            )
            for name in request.plan.registration_names
        }
        access_sha = _sha256_bytes(
            _canonical_json_bytes(
                {
                    "inputs": inputs,
                    "stage12_binding_sha256": request.stage12_binding_sha256,
                }
            )
        )
        output_sha = _sha256_bytes(
            _canonical_json_bytes(
                {
                    "access_receipt_sha256": access_sha,
                    "previous_state_sha256": request.previous_state_sha256,
                    "execution_config_sha256": request.execution_config_sha256,
                }
            )
        )
        return FrozenMonthOutput(
            input_sha256_by_registration=inputs,
            access_receipt_sha256=access_sha,
            output_sha256=output_sha,
            runtime_id=self.runtime_id,
            synthetic=True,
        )


def _checkpoint_state(
    request: FrozenMonthRequest,
    output: FrozenMonthOutput,
) -> bytes:
    output.validate_for(request)
    return _canonical_json_bytes(
        {
            "schema_version": 1,
            "segment": request.plan.segment.value,
            "chain_id": request.plan.chain_id,
            "holding_month": request.holding_month,
            "registration_names": list(request.plan.registration_names),
            "stage12_binding_sha256": request.stage12_binding_sha256,
            "execution_config_sha256": request.execution_config_sha256,
            "previous_state_sha256": request.previous_state_sha256,
            "input_sha256_by_registration": dict(
                sorted(output.input_sha256_by_registration.items())
            ),
            "access_receipt_sha256": output.access_receipt_sha256,
            "output_sha256": output.output_sha256,
            "runtime_id": output.runtime_id,
            "synthetic": output.synthetic,
        }
    )


def _authenticate_resumed_state(
    loaded: LoadedCheckpoint,
    *,
    plan: FrozenSegmentPlan,
    binding: Stage12VerifiedBinding,
    execution_config_sha256: str,
    synthetic: bool,
) -> None:
    try:
        state = json.loads(loaded.state)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Stage13ExecutionError("resumed checkpoint state is invalid") from exc
    if not isinstance(state, Mapping) or (
        state.get("schema_version") != 1
        or state.get("segment") != plan.segment.value
        or state.get("chain_id") != plan.chain_id
        or state.get("holding_month") != loaded.record.holding_month
        or state.get("registration_names") != list(plan.registration_names)
        or state.get("stage12_binding_sha256") != binding.binding_sha256
        or state.get("execution_config_sha256") != execution_config_sha256
        or state.get("synthetic") is not synthetic
    ):
        raise Stage13ExecutionError("resumed checkpoint semantics differ")


def _run_stage13_segments(
    *,
    checkpoint_root: Path,
    binding: Stage12VerifiedBinding,
    runtime: _SyntheticFrozenRuntime,
) -> Stage13RunResult:
    """Private synthetic-only runner; production never accepts this injection."""

    if not isinstance(runtime, _SyntheticFrozenRuntime):
        raise Stage13ExecutionError("only the private synthetic runtime is injectable")
    if not isinstance(checkpoint_root, Path) or not checkpoint_root.is_absolute():
        raise Stage13ExecutionError("synthetic checkpoint root must be absolute")
    execution_sha = frozen_execution_config_sha256()
    segment_results: list[SegmentExecutionResult] = []
    for segment_plan in FROZEN_SEGMENT_PLANS:
        checkpoint_plan = CheckpointPlan(
            chain_id=segment_plan.chain_id,
            stage12_binding=binding,
            execution_config_sha256=execution_sha,
            holding_months=segment_plan.holding_months,
        )
        store = MonthlyCheckpointStore(
            checkpoint_root / segment_plan.segment.value,
            checkpoint_plan,
        )
        latest = store.resume_from_latest()
        resumed = 0 if latest is None else latest.record.ordinal
        if latest is not None:
            _authenticate_resumed_state(
                latest,
                plan=segment_plan,
                binding=binding,
                execution_config_sha256=execution_sha,
                synthetic=True,
            )
        executed: list[str] = []
        previous_state_sha = None if latest is None else latest.record.state_sha256
        for month in segment_plan.holding_months[resumed:]:
            request = FrozenMonthRequest(
                plan=segment_plan,
                holding_month=month,
                stage12_binding_sha256=binding.binding_sha256,
                execution_config_sha256=execution_sha,
                previous_state_sha256=previous_state_sha,
            )
            output = runtime.execute(request)
            latest = store.commit(
                holding_month=month,
                state=_checkpoint_state(request, output),
                state_format=STAGE13_STATE_FORMAT,
            )
            previous_state_sha = latest.record.state_sha256
            executed.append(month)
        if latest is None or latest.record.ordinal != len(segment_plan.holding_months):
            raise Stage13ExecutionError("segment chain did not reach its exact head")
        reversed_chain: list[MonthlyCheckpointRecord] = [latest.record]
        current = latest.record
        while current.previous_checkpoint_sha256 is not None:
            previous = store._load_record(  # noqa: SLF001 - authenticated store API
                current.previous_checkpoint_sha256,
                load_state=False,
            )
            if not isinstance(previous, MonthlyCheckpointRecord):
                raise Stage13ExecutionError("checkpoint ancestry loader differed")
            reversed_chain.append(previous)
            current = previous
        chain = tuple(reversed(reversed_chain))
        segment_results.append(
            SegmentExecutionResult(
                plan=segment_plan,
                checkpoint_plan_sha256=checkpoint_plan.plan_sha256,
                head=latest.record,
                chain_records=chain,
                executed_months=tuple(executed),
                resumed_month_count=resumed,
                synthetic=True,
            )
        )
    return Stage13RunResult(
        stage12_binding=binding,
        execution_config_sha256=execution_sha,
        segments=tuple(segment_results),
        synthetic=True,
    )


def run_stage13_final_test() -> object:
    """Run the only production Stage 13 path after all fixed authorities pass.

    Stage 12 technical readiness does not itself authorize the unblind.  This
    zero-argument entry authenticates the Stage 12 verified pointer first,
    then the repository-fixed Stage 13 approval, then the repository-fixed
    verified registration pointer.  Only the resulting broker can reach the
    native production runtime.
    """

    from portfolio_bot.final_test.authorization import (
        Stage13AuthorizationError,
        load_authenticated_stage12_binding,
        require_stage13_unblind_authorization,
    )
    from portfolio_bot.final_test.input_materialization import (
        FinalInputMaterializationError,
        load_verified_final_access_broker,
    )
    from portfolio_bot.final_test.final_input_construction import (
        FinalInputConstructionError,
        construct_and_publish_final_inputs,
    )
    from portfolio_bot.final_test.production_runtime import (
        Stage13ProductionRuntimeError,
        _stable_fixed_approval_receipt,
        run_authenticated_stage13_execution,
    )

    try:
        binding = load_authenticated_stage12_binding(PROJECT_ROOT)
        approval = _stable_fixed_approval_receipt()
        approval = require_stage13_unblind_authorization(
            approval, expected_stage12_binding=binding
        )
        # Construction is itself authorization-first and memoized.  Calling it
        # here ensures the verified registration is derived from authenticated
        # normalized sources, rather than assuming somebody populated a pointer.
        construct_and_publish_final_inputs()
        broker = load_verified_final_access_broker(PROJECT_ROOT, approval)
        return run_authenticated_stage13_execution(
            binding=binding,
            approval=approval,
            broker=broker,
        )
    except (
        Stage13AuthorizationError,
        FinalInputConstructionError,
        FinalInputMaterializationError,
        Stage13ProductionRuntimeError,
    ) as exc:
        if isinstance(exc, Stage13ExecutionError):
            raise
        raise Stage13ExecutionError(str(exc)) from exc


__all__ = [
    "ECONOMIC_RUNTIME_STATUS",
    "FROZEN_SEGMENT_PLANS",
    "FrozenRegistrationManifestEntry",
    "REGISTRATION_RUNTIME_STATUS",
    "SegmentExecutionResult",
    "Stage13ExecutionError",
    "Stage13RunResult",
    "Stage13Segment",
    "frozen_execution_config_sha256",
    "run_stage13_final_test",
]
