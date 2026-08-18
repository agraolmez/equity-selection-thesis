"""Native-free execution controller for the frozen Stage 9 selector search.

The parent process represented by this module never imports a model framework.
It authenticates prepared data, creates content-addressed ``NativeFitSpec``
jobs, drives one long-lived ``spawn`` worker per chronological fold, reloads
every completed cache artifact, and performs development-only OOF evaluation.
XGBoost's 40-config core is sealed before its eight gamma challengers are even
constructible.  The final handoff is the immutable search bundle writer; this
module does not publish a primary pointer or authorize Stage 10.
"""

from __future__ import annotations

import hashlib
import json
import math
import multiprocessing as mp
import os
import queue
import re
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np
import pandas as pd

from portfolio_bot.selectors.cache import (
    atomic_write_json,
    canonical_json_bytes,
    job_id_from_identity,
    load_completed_job,
    sha256_file,
)
from portfolio_bot.selectors.complexity import (
    elastic_net_complexity,
    neural_complexity,
    xgboost_complexity,
)
from portfolio_bot.selectors.contract import (
    CONTRACT_ID,
    DEFAULT_POLICY,
    validate_contract,
    validate_runtime_versions,
)
from portfolio_bot.selectors.evaluation import (
    FamilyEvaluation,
    evaluate_family_predictions,
)
from portfolio_bot.selectors.fit_handlers import (
    family_code_sha256,
    prepared_input_bundle_sha256,
    support_binding_sha256,
)
from portfolio_bot.selectors.model_specs import (
    DEVELOPMENT_SEEDS,
    ELASTIC_NET_PATH_CONFIG_ID,
    NeuralConfig,
    XGBoostConfig,
    configuration_payload_sha256,
    core_xgboost_configurations,
    elastic_net_configurations,
    elastic_net_path_payload_sha256,
    neural_configurations,
)
from portfolio_bot.selectors.native_worker import (
    RUNTIME_VERSIONS_SHA256,
    NativeFitSpec,
    run_native_family_shard,
)
from portfolio_bot.selectors.prepared_data import (
    EXPECTED_FOLDS,
    PreparedSelectorData,
    load_prepared_selector_data,
)
from portfolio_bot.selectors.results_bundle import (
    OOFDomain,
    assemble_elastic_cube,
    assemble_stochastic_cube,
    build_oof_domain,
)
from portfolio_bot.selectors.search_bundle import (
    CompletedSearchInputs,
    CompletedSearchPublication,
    publish_completed_search_bundle,
)
from portfolio_bot.selectors.search_plan import (
    derive_gamma_challengers,
    frozen_search_plan_counts,
    select_gamma_parent_ids,
)


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_FAMILIES = ("neural_network", "xgboost", "elastic_net")
_HANDLER_MODULE = "portfolio_bot.selectors.fit_handlers"
_HANDLER_FACTORY = "create_fold_fit_handler"
_DEFAULT_HEARTBEAT_SECONDS = 30.0
_ELASTIC_NET_FOLD1_RUNTIME_LIMIT_SECONDS = 1_800.0
_APFS_STABLE_READ_ATTEMPTS = 3


class Stage9SearchExecutionError(RuntimeError):
    """Raised when an authoritative search phase cannot finish safely."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Stage9SearchExecutionError(message)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _array_sha256(values: np.ndarray, *, chunk_rows: int = 16_384) -> str:
    array = np.asarray(values)
    digest = hashlib.sha256()
    digest.update(canonical_json_bytes({"dtype": array.dtype.str, "shape": list(array.shape)}))
    for start in range(0, array.shape[0], chunk_rows):
        block = np.ascontiguousarray(array[start : start + chunk_rows])
        digest.update(memoryview(block).cast("B"))
    return digest.hexdigest()


def _load_completed_job_with_bounded_retry(
    path: Path, **kwargs: Any
) -> dict[str, Any] | None:
    """Require one stable authenticated read, with bounded APFS settlement retries."""

    for _attempt in range(_APFS_STABLE_READ_ATTEMPTS):
        receipt = load_completed_job(path, **kwargs)
        if receipt is not None:
            return receipt
    return None


def _elastic_net_fold1_deadline(
    plan: "NativePhasePlan",
    batch: Sequence[str],
    *,
    started_at: float,
) -> float | None:
    """Return the end-to-end Elastic Net fold-1 cold-fit deadline."""

    if plan.family == "elastic_net" and tuple(batch) == (EXPECTED_FOLDS[0],):
        return float(started_at) + _ELASTIC_NET_FOLD1_RUNTIME_LIMIT_SECONDS
    return None


def _bounded_result_wait_seconds(
    heartbeat_seconds: float,
    *,
    deadline: float | None,
    now: float,
) -> float:
    """Bound a result-queue wait so the fold-1 runtime gate cannot be skipped."""

    heartbeat = float(heartbeat_seconds)
    _require(math.isfinite(heartbeat) and heartbeat > 0.0, "Native phase heartbeat must be positive")
    if deadline is None:
        return heartbeat
    remaining = float(deadline) - float(now)
    _require(
        math.isfinite(remaining) and remaining > 0.0,
        "Elastic Net fold-1 preflight exceeded its configured runtime deadline",
    )
    return min(heartbeat, remaining)


def _require_cold_fold1_completion(
    completion: Mapping[str, Any],
    *,
    deadline: float | None,
    allow_authenticated_cache: bool = False,
) -> None:
    """Enforce cold fold 1, except for an explicit authenticated resume."""

    if deadline is not None:
        if allow_authenticated_cache:
            _require(
                completion.get("cache_hit") is True,
                "Elastic Net partial resume requires an existing authenticated "
                "fold-1 parent cache",
            )
        else:
            _require(
                completion.get("cache_hit") is False,
                "Elastic Net fold-1 preflight requires a cold parent fit; an existing "
                "parent cache cannot satisfy its runtime gate without explicit resume",
            )


@dataclass(frozen=True)
class NativePhasePlan:
    phase_id: str
    family: str
    max_parallel_folds: int
    specifications_by_fold: Mapping[str, tuple[NativeFitSpec, ...]]
    initialization_by_fold: Mapping[str, Mapping[str, Any]]

    @property
    def specifications(self) -> tuple[NativeFitSpec, ...]:
        return tuple(
            specification
            for fold_id in EXPECTED_FOLDS
            for specification in self.specifications_by_fold[fold_id]
        )

    @property
    def job_count(self) -> int:
        return sum(len(value) for value in self.specifications_by_fold.values())

    def validated(self) -> "NativePhasePlan":
        _require(self.family in _FAMILIES, "Native phase family differs")
        _require(
            tuple(self.specifications_by_fold) == EXPECTED_FOLDS
            and tuple(self.initialization_by_fold) == EXPECTED_FOLDS,
            "Native phase fold order differs",
        )
        _require(self.job_count > 0, "Native phase contains no jobs")
        _require(
            1 <= int(self.max_parallel_folds) <= len(EXPECTED_FOLDS),
            "Native phase fold parallelism differs",
        )
        if self.family == "elastic_net":
            _require(
                self.max_parallel_folds == 1,
                "Elastic Net folds must execute sequentially within the 32GB memory budget",
            )
        identities: set[str] = set()
        for fold_id in EXPECTED_FOLDS:
            specifications = self.specifications_by_fold[fold_id]
            _require(specifications, f"Native phase omitted {fold_id}")
            for specification in specifications:
                specification.validated()
                _require(
                    specification.family == self.family
                    and specification.fold_id == fold_id,
                    "Native phase specification escaped its family/fold",
                )
                job_id = job_id_from_identity(specification.job_identity())
                _require(job_id not in identities, "Native phase job identity is duplicated")
                identities.add(job_id)
        return self


@dataclass(frozen=True)
class NativePhaseResult:
    phase_id: str
    family: str
    job_count: int
    cache_hits: int
    elapsed_seconds: float
    completions: Mapping[str, Mapping[str, Any]]


@dataclass(frozen=True)
class PhaseArtifacts:
    predictions: Mapping[tuple[str, str, int | None], np.ndarray]
    publication_fit_receipts: tuple[Mapping[str, Any], ...]
    cache_sidecar_sha256s: Mapping[str, str]


@dataclass(frozen=True)
class SealedCoreReceipt:
    core_search_id: str
    path: Path
    receipt: Mapping[str, Any]
    sha256: str
    gamma_parent_config_ids: tuple[str, ...]


@dataclass(frozen=True)
class Stage9SearchExecution:
    publication: CompletedSearchPublication
    sealed_core: SealedCoreReceipt
    phase_results: tuple[NativePhaseResult, ...]
    evaluated_configurations: int
    fit_receipts: int
    cache_hits: int
    elapsed_seconds: float


Progress = Callable[[str], None]


class PhaseExecutor(Protocol):
    def __call__(
        self,
        plan: NativePhasePlan,
        *,
        progress: Progress,
        heartbeat_seconds: float,
    ) -> NativePhaseResult: ...


def _safe_child_roots(
    database_root: Path,
    policy: Mapping[str, Any],
    prepared_build_id: str,
) -> Path:
    relative = Path(str(policy["database"]["cache_root"]))
    _require(
        relative.parts and not relative.is_absolute() and ".." not in relative.parts,
        "Stage 9 cache root is unsafe",
    )
    root = database_root / relative / prepared_build_id
    try:
        root.relative_to(database_root)
    except ValueError as exc:  # pragma: no cover - guarded above
        raise Stage9SearchExecutionError("Stage 9 cache escaped database root") from exc
    return root


def _create_safe_directory(path: Path, *, trusted_root: Path) -> None:
    path = Path(os.path.abspath(path))
    try:
        path.relative_to(trusted_root)
    except ValueError as exc:
        raise Stage9SearchExecutionError("Worker directory escaped database root") from exc
    current = trusted_root
    for part in path.relative_to(trusted_root).parts:
        current = current / part
        if current.exists() or current.is_symlink():
            _require(
                current.is_dir() and not current.is_symlink(),
                f"Unsafe worker directory component: {current}",
            )
        else:
            current.mkdir(mode=0o700)


def _support_sha256(
    prepared: PreparedSelectorData,
    *,
    family: str,
    fold_id: str,
) -> str:
    if family in {"neural_network", "xgboost"}:
        return support_binding_sha256(
            family,
            fold_id,
            inner_base_support=prepared.inner_base_support[fold_id],
            full_base_support=prepared.full_base_support[fold_id],
        )
    return support_binding_sha256(
        family,
        fold_id,
        elastic_support=prepared.elastic_support[fold_id],
    )


def _config_identity(
    family: str,
    config: NeuralConfig | XGBoostConfig | str,
) -> tuple[str, str, tuple[int | None, ...]]:
    if family == "neural_network":
        assert isinstance(config, NeuralConfig)
        return config.config_id, configuration_payload_sha256(config), DEVELOPMENT_SEEDS
    if family == "xgboost":
        assert isinstance(config, XGBoostConfig)
        return config.config_id, configuration_payload_sha256(config), DEVELOPMENT_SEEDS
    _require(config == ELASTIC_NET_PATH_CONFIG_ID, "Elastic Net phase config differs")
    return ELASTIC_NET_PATH_CONFIG_ID, elastic_net_path_payload_sha256(), (None,)


def build_native_phase_plan(
    prepared: PreparedSelectorData,
    *,
    family: str,
    phase_id: str,
    configurations: Sequence[NeuralConfig | XGBoostConfig | str],
    database_root: Path,
    policy_path: Path = DEFAULT_POLICY,
    gamma_authorization: Mapping[str, Any] | None = None,
) -> NativePhasePlan:
    """Create the complete content-addressed job ledger for one native phase."""

    policy, _ = validate_contract(policy_path)
    _require(family in _FAMILIES, "Unknown native phase family")
    _require(configurations, "Native phase configuration list is empty")
    if gamma_authorization is not None:
        _require(family == "xgboost", "Only XGBoost may carry gamma authorization")
    database_root = Path(os.path.abspath(database_root.expanduser()))
    _require(
        database_root.is_dir()
        and not database_root.is_symlink()
        and str(database_root) == str(policy["database"]["root"]),
        "Native phase database root differs from the frozen policy",
    )
    base = _safe_child_roots(database_root, policy, str(prepared.metadata["build_id"]))
    rules_sha = str(policy["physical_rules"]["sha256"])
    input_sha = prepared_input_bundle_sha256(prepared.metadata)
    code_sha = family_code_sha256(family)
    specs_by_fold: dict[str, tuple[NativeFitSpec, ...]] = {}
    initialization: dict[str, Mapping[str, Any]] = {}
    for fold_id in EXPECTED_FOLDS:
        cache_root = base / "fit_jobs" / phase_id / fold_id
        scratch_root = base / "scratch" / phase_id / fold_id
        _create_safe_directory(cache_root, trusted_root=database_root)
        if family == "elastic_net":
            _create_safe_directory(scratch_root, trusted_root=database_root)
        support_sha = _support_sha256(prepared, family=family, fold_id=fold_id)
        specifications: list[NativeFitSpec] = []
        for configuration in configurations:
            config_id, config_sha, seeds = _config_identity(family, configuration)
            for seed in seeds:
                provisional = NativeFitSpec(
                    family=family,
                    config_id=config_id,
                    fold_id=fold_id,
                    seed=seed,
                    config_payload_sha256=config_sha,
                    runtime_versions_sha256=RUNTIME_VERSIONS_SHA256,
                    rules_sha256=rules_sha,
                    input_bundle_sha256=input_sha,
                    code_sha256=code_sha,
                    support_sha256=support_sha,
                    prepared_root=str(prepared.root),
                    cache_path=str((cache_root / "provisional.npz").resolve()),
                    sealed_core_receipt_sha256=(
                        None
                        if gamma_authorization is None
                        else str(gamma_authorization["sealed_core_receipt_sha256"])
                    ),
                )
                identity = provisional.job_identity()
                cache_path = cache_root / f"{job_id_from_identity(identity)}.npz"
                specifications.append(replace(provisional, cache_path=str(cache_path)))
        specs_by_fold[fold_id] = tuple(specifications)
        initialization[fold_id] = {
            "family": family,
            "fold_id": fold_id,
            "prepared_root": str(prepared.root),
            "expected_build_id": prepared.metadata["build_id"],
            "expected_content_fingerprint": prepared.metadata["content_fingerprint"],
            "cache_root": str(cache_root),
            "scratch_root": str(scratch_root) if family == "elastic_net" else None,
            "rules_sha256": rules_sha,
            "gamma_authorization": (
                None if gamma_authorization is None else dict(gamma_authorization)
            ),
        }
    return NativePhasePlan(
        phase_id=phase_id,
        family=family,
        max_parallel_folds=1 if family == "elastic_net" else 4,
        specifications_by_fold=specs_by_fold,
        initialization_by_fold=initialization,
    ).validated()


def _stop_processes(
    processes: Mapping[str, mp.Process],
    request_queues: Mapping[str, Any],
    *,
    terminate: bool,
) -> None:
    for fold_id, process in processes.items():
        if process.is_alive():
            if terminate:
                process.terminate()
            else:
                request_queues[fold_id].put({"command": "STOP"})
    for process in processes.values():
        process.join(timeout=30.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=10.0)
    for value in request_queues.values():
        value.close()


def execute_native_phase(
    plan: NativePhasePlan,
    *,
    progress: Progress,
    heartbeat_seconds: float = _DEFAULT_HEARTBEAT_SECONDS,
    allow_authenticated_elastic_net_fold1_cache: bool = False,
) -> NativePhaseResult:
    """Drive four fold-isolated spawn workers until every job is complete."""

    plan.validated()
    _require(
        not allow_authenticated_elastic_net_fold1_cache
        or (
            plan.family == "elastic_net"
            and plan.max_parallel_folds == 1
            and plan.job_count == len(EXPECTED_FOLDS)
        ),
        "Authenticated fold-1 resume is restricted to the canonical four-parent "
        "sequential Elastic Net phase",
    )
    _require(
        math.isfinite(float(heartbeat_seconds)) and heartbeat_seconds > 0.0,
        "Native phase heartbeat must be positive",
    )
    started = time.monotonic()
    context = mp.get_context("spawn")
    completions: dict[str, Mapping[str, Any]] = {}
    planned_job_ids = {
        job_id_from_identity(specification.job_identity())
        for specification in plan.specifications
    }
    _require(
        len(planned_job_ids) == plan.job_count,
        "Native phase planned job set differs",
    )
    cache_hits = 0
    fold_batches = [
        EXPECTED_FOLDS[start : start + plan.max_parallel_folds]
        for start in range(0, len(EXPECTED_FOLDS), plan.max_parallel_folds)
    ]
    for batch_number, batch in enumerate(fold_batches, start=1):
        batch_started_at = time.monotonic()
        deadline = _elastic_net_fold1_deadline(
            plan,
            batch,
            started_at=batch_started_at,
        )
        result_queue = context.Queue()
        request_queues: dict[str, Any] = {}
        processes: dict[str, mp.Process] = {}
        batch_jobs = sum(len(plan.specifications_by_fold[fold]) for fold in batch)
        batch_completions = 0
        try:
            for fold_id in batch:
                requests = context.Queue()
                process = context.Process(
                    target=run_native_family_shard,
                    kwargs={
                        "family": plan.family,
                        "handler_module": _HANDLER_MODULE,
                        "handler_factory": _HANDLER_FACTORY,
                        "initialization_payload": dict(plan.initialization_by_fold[fold_id]),
                        "request_queue": requests,
                        "result_queue": result_queue,
                    },
                    name=f"stage09-{plan.phase_id}-{fold_id}",
                )
                process.start()
                request_queues[fold_id] = requests
                processes[fold_id] = process

            ready: set[str] = set()
            while len(ready) < len(batch):
                try:
                    wait_seconds = _bounded_result_wait_seconds(
                        heartbeat_seconds,
                        deadline=deadline,
                        now=time.monotonic(),
                    )
                    message = result_queue.get(timeout=wait_seconds)
                except queue.Empty:
                    if deadline is not None:
                        _bounded_result_wait_seconds(
                            heartbeat_seconds,
                            deadline=deadline,
                            now=time.monotonic(),
                        )
                    dead = [fold for fold, process in processes.items() if not process.is_alive()]
                    _require(not dead, f"Native phase workers died during initialization: {dead}")
                    progress(
                        f"{plan.phase_id}: initializing batch {batch_number}/{len(fold_batches)} "
                        f"({len(ready)}/{len(batch)} workers ready)"
                    )
                    continue
                if deadline is not None:
                    _bounded_result_wait_seconds(
                        heartbeat_seconds,
                        deadline=deadline,
                        now=time.monotonic(),
                    )
                _require(message.get("status") == "READY", f"Native phase initialization failed: {message}")
                ready.add(str(message["pid"]))

            for fold_id in batch:
                for specification in plan.specifications_by_fold[fold_id]:
                    request_queues[fold_id].put(specification)
            progress(
                f"{plan.phase_id}: dispatched batch {batch_number}/{len(fold_batches)} "
                f"({batch_jobs} jobs; folds {', '.join(batch)})"
            )
            if deadline is not None:
                gate_mode = (
                    "authenticated-cache resume deadline"
                    if allow_authenticated_elastic_net_fold1_cache
                    else "cold end-to-end runtime gate"
                )
                progress(
                    f"{plan.phase_id}: dev_fold_1 {gate_mode} active "
                    f"({_ELASTIC_NET_FOLD1_RUNTIME_LIMIT_SECONDS:.0f} seconds)"
                )

            while batch_completions < batch_jobs:
                try:
                    wait_seconds = _bounded_result_wait_seconds(
                        heartbeat_seconds,
                        deadline=deadline,
                        now=time.monotonic(),
                    )
                    message = result_queue.get(timeout=wait_seconds)
                except queue.Empty:
                    if deadline is not None:
                        _bounded_result_wait_seconds(
                            heartbeat_seconds,
                            deadline=deadline,
                            now=time.monotonic(),
                        )
                    dead = [
                        (fold, process.exitcode)
                        for fold, process in processes.items()
                        if not process.is_alive()
                    ]
                    _require(not dead, f"Native phase worker exited before completion: {dead}")
                    progress(
                        f"{plan.phase_id}: {len(completions)}/{plan.job_count} jobs complete; "
                        "workers still active"
                    )
                    continue
                if deadline is not None:
                    _bounded_result_wait_seconds(
                        heartbeat_seconds,
                        deadline=deadline,
                        now=time.monotonic(),
                    )
                status = message.get("status")
                _require(status != "ERROR", f"Native phase worker error: {message}")
                if status != "COMPLETE":
                    continue
                identity = dict(message["job_identity"])
                job_id = job_id_from_identity(identity)
                completion = dict(message["receipt"])
                _require_cold_fold1_completion(
                    completion,
                    deadline=deadline,
                    allow_authenticated_cache=(
                        allow_authenticated_elastic_net_fold1_cache
                    ),
                )
                _require(
                    completion.get("status") == "COMPLETE"
                    and completion.get("job_id") == job_id
                    and job_id in planned_job_ids
                    and job_id not in completions,
                    "Native phase returned an unplanned, malformed, or duplicate completion",
                )
                completions[job_id] = completion
                batch_completions += 1
                cache_hits += int(bool(completion.get("cache_hit")))
                completed = len(completions)
                if completed == plan.job_count or completed % max(1, plan.job_count // 20) == 0:
                    progress(
                        f"{plan.phase_id}: {completed}/{plan.job_count} jobs complete "
                        f"({cache_hits} authenticated cache hits)"
                    )
            _stop_processes(processes, request_queues, terminate=False)
            bad = {fold: process.exitcode for fold, process in processes.items() if process.exitcode != 0}
            _require(not bad, f"Native phase workers exited unsuccessfully: {bad}")
        except BaseException:
            _stop_processes(processes, request_queues, terminate=True)
            raise
        finally:
            result_queue.close()
    _require(
        set(completions) == planned_job_ids,
        "Native phase completed job set differs from its plan",
    )
    return NativePhaseResult(
        phase_id=plan.phase_id,
        family=plan.family,
        job_count=plan.job_count,
        cache_hits=cache_hits,
        elapsed_seconds=float(time.monotonic() - started),
        completions=completions,
    )


def load_authenticated_cached_phase(
    plan: NativePhasePlan,
    prepared: PreparedSelectorData,
    *,
    progress: Progress,
) -> NativePhaseResult:
    """Load a complete native phase from cache without launching any worker.

    This is the fail-closed resume path used after a prior cold Elastic Net
    phase completed.  Every planned artifact must authenticate against its
    exact identity and frozen outer-fold shape before a result is returned;
    a missing or invalid cache raises immediately and can never fall back to
    model fitting.
    """

    started = time.monotonic()
    plan.validated()
    _require(
        plan.family == "elastic_net" and plan.job_count == len(EXPECTED_FOLDS),
        "Authenticated cache-only resume is restricted to the four-fold Elastic Net phase",
    )
    outer_rows = {
        fold.fold_id: int(fold.outer_validation.size) for fold in prepared.folds
    }
    _require(
        set(outer_rows) == set(EXPECTED_FOLDS),
        "Authenticated resume outer-fold domain differs",
    )
    completions: dict[str, Mapping[str, Any]] = {}
    for specification in plan.specifications:
        identity = specification.job_identity()
        job_id = job_id_from_identity(identity)
        cache_path = Path(specification.cache_path)
        receipt = _load_completed_job_with_bounded_retry(
            cache_path,
            trusted_root=cache_path.parent,
            expected_identity=identity,
            expected_array_schema=specification.expected_array_schema(
                outer_rows[specification.fold_id]
            ),
        )
        _require(
            receipt is not None,
            f"Authenticated cache-only phase is missing or invalid: {job_id}",
        )
        completion = {
            "schema_version": 1,
            "status": "COMPLETE",
            "cache_hit": True,
            "job_id": receipt["job_id"],
            "job_identity_sha256": receipt["job_identity_sha256"],
            "prediction_sha256": receipt["prediction_sha256"],
            "fit_receipt_sha256": receipt["fit_receipt_sha256"],
            "cache_path": str(cache_path),
        }
        _require(
            job_id not in completions and completion["job_id"] == job_id,
            "Authenticated cache-only phase returned a duplicate identity",
        )
        completions[job_id] = completion
        progress(
            f"{plan.phase_id}: {len(completions)}/{plan.job_count} jobs complete "
            f"({len(completions)} authenticated cache hits; no workers launched)"
        )
    _require(
        len(completions) == plan.job_count,
        "Authenticated cache-only phase is incomplete",
    )
    return NativePhaseResult(
        phase_id=plan.phase_id,
        family=plan.family,
        job_count=plan.job_count,
        cache_hits=plan.job_count,
        elapsed_seconds=float(time.monotonic() - started),
        completions=completions,
    )


def load_phase_artifacts(
    plan: NativePhasePlan,
    result: NativePhaseResult,
    prepared: PreparedSelectorData,
) -> PhaseArtifacts:
    """Reauthenticate every job sidecar and load only finite float32 forecasts."""

    _require(
        result.phase_id == plan.phase_id
        and result.family == plan.family
        and result.job_count == plan.job_count
        and len(result.completions) == plan.job_count,
        "Native phase result does not match its plan",
    )
    outer_rows = {
        fold.fold_id: int(fold.outer_validation.size) for fold in prepared.folds
    }
    predictions: dict[tuple[str, str, int | None], np.ndarray] = {}
    publication_rows: list[Mapping[str, Any]] = []
    sidecar_hashes: dict[str, str] = {}
    for specification in plan.specifications:
        identity = specification.job_identity()
        job_id = job_id_from_identity(identity)
        completion = result.completions.get(job_id)
        _require(completion is not None, f"Missing native phase completion: {job_id}")
        cache_path = Path(specification.cache_path)
        schema = specification.expected_array_schema(outer_rows[specification.fold_id])
        receipt = _load_completed_job_with_bounded_retry(
            cache_path,
            trusted_root=cache_path.parent,
            expected_identity=identity,
            expected_array_schema=schema,
        )
        _require(receipt is not None, f"Completed job failed cache authentication: {job_id}")
        _require(
            completion.get("prediction_sha256") == receipt["prediction_sha256"]
            and completion.get("fit_receipt_sha256") == receipt["fit_receipt_sha256"],
            f"Worker/cache completion binding differs: {job_id}",
        )
        with np.load(cache_path, allow_pickle=False) as archive:
            forecast = np.asarray(archive["predictions"], dtype=np.float32).copy()
        _require(
            forecast.dtype == np.dtype("float32") and np.isfinite(forecast).all(),
            f"Cached forecast is invalid: {job_id}",
        )
        # Authenticate again after reading to close the read/verify race.
        after = _load_completed_job_with_bounded_retry(
            cache_path,
            trusted_root=cache_path.parent,
            expected_identity=identity,
            expected_array_schema=schema,
        )
        _require(after == receipt, f"Completed job changed while loading: {job_id}")
        fit_receipt = receipt.get("fit_receipt")
        _require(isinstance(fit_receipt, Mapping), f"Fit receipt is missing: {job_id}")
        normalized = fit_receipt.get("publication_fit_receipts")
        expected_rows = 250 if plan.family == "elastic_net" else 1
        _require(
            isinstance(normalized, list) and len(normalized) == expected_rows,
            f"Publication fit ledger differs: {job_id}",
        )
        normalized_rows = [dict(value) for value in normalized]
        if plan.family == "elastic_net":
            native_evidence = fit_receipt.get("model_fit")
            _require(
                isinstance(native_evidence, list)
                and len(native_evidence) == expected_rows,
                f"Elastic Net native fit ledger differs: {job_id}",
            )
            gap_limits: dict[str, float] = {}
            for value in native_evidence:
                _require(
                    isinstance(value, Mapping)
                    and isinstance(value.get("config_id"), str),
                    f"Elastic Net native certificate identity differs: {job_id}",
                )
                config_id = str(value["config_id"])
                limit = float(value.get("dual_gap_limit", math.nan))
                _require(
                    config_id not in gap_limits
                    and math.isfinite(limit)
                    and limit >= 0.0,
                    f"Elastic Net native dual-gap limit differs: {job_id}",
                )
                gap_limits[config_id] = limit
            _require(
                set(gap_limits)
                == {str(value.get("config_id")) for value in normalized_rows},
                f"Elastic Net normalized/native certificate ledgers differ: {job_id}",
            )
            for value in normalized_rows:
                value["dual_gap_limit"] = gap_limits[str(value["config_id"])]
        publication_rows.extend(normalized_rows)
        key = (specification.fold_id, specification.config_id, specification.seed)
        _require(key not in predictions, f"Loaded prediction identity is duplicated: {key}")
        predictions[key] = forecast
        sidecar_hashes[job_id] = sha256_file(cache_path.with_suffix(".json"))
    return PhaseArtifacts(
        predictions=predictions,
        publication_fit_receipts=tuple(publication_rows),
        cache_sidecar_sha256s=dict(sorted(sidecar_hashes.items())),
    )


def _stochastic_cube(
    domain: OOFDomain,
    prepared: PreparedSelectorData,
    artifacts: PhaseArtifacts,
    *,
    config_ids: Sequence[str],
) -> np.ndarray:
    return assemble_stochastic_cube(
        domain,
        prepared.folds,
        config_ids=config_ids,
        prediction_loader=lambda fold_id, config_id, seed: artifacts.predictions[
            (fold_id, config_id, seed)
        ],
    )


def _elastic_cube(
    domain: OOFDomain,
    prepared: PreparedSelectorData,
    artifacts: PhaseArtifacts,
) -> np.ndarray:
    config_ids = tuple(value.config_id for value in elastic_net_configurations())
    return assemble_elastic_cube(
        domain,
        prepared.folds,
        config_ids=config_ids,
        fold_prediction_loader=lambda fold_id: artifacts.predictions[
            (fold_id, ELASTIC_NET_PATH_CONFIG_ID, None)
        ],
    )


def seal_xgboost_core_receipt(
    *,
    prepared: PreparedSelectorData,
    core_cube: np.ndarray,
    core_evaluation: FamilyEvaluation,
    core_complexity: pd.DataFrame,
    artifacts: PhaseArtifacts,
    database_root: Path,
    policy_path: Path = DEFAULT_POLICY,
) -> SealedCoreReceipt:
    """Seal core metrics and appoint exactly four gamma parents by frozen rule."""

    policy, _ = validate_contract(policy_path)
    core_ids = tuple(value.config_id for value in core_xgboost_configurations())
    _require(
        core_evaluation.model_family == "xgboost"
        and core_evaluation.config_ids == core_ids
        and np.asarray(core_cube).shape[1:] == (40, 3),
        "XGBoost core evaluation/cube differs",
    )
    metrics = core_evaluation.config_metrics.merge(
        core_complexity[["config_id", "complexity"]],
        on="config_id",
        validate="one_to_one",
    )
    parents = select_gamma_parent_ids(
        metrics[["config_id", "mean_rank_ic", "complexity"]]
    )
    metric_records = [
        {
            "config_id": str(row.config_id),
            "mean_rank_ic": float(row.mean_rank_ic),
            "complexity": float(row.complexity),
        }
        for row in metrics.sort_values("config_id", kind="mergesort").itertuples(index=False)
    ]
    identity_payload = {
        "schema_version": 1,
        "purpose": "stage09_sealed_xgboost_core_before_gamma",
        "contract_id": CONTRACT_ID,
        "prepared_build_id": prepared.metadata["build_id"],
        "prepared_content_fingerprint": prepared.metadata["content_fingerprint"],
        "rules_sha256": policy["physical_rules"]["sha256"],
        "core_config_ids": list(core_ids),
        "core_oof_sha256": _array_sha256(np.asarray(core_cube, dtype=np.float32)),
        "core_metrics": metric_records,
        "fit_cache_sidecar_sha256s": dict(artifacts.cache_sidecar_sha256s),
        "gamma_parent_config_ids": list(parents),
    }
    fingerprint = _sha256_bytes(canonical_json_bytes(identity_payload))
    core_id = f"stage09-xgboost-core-{fingerprint[:24]}"
    receipt = {
        **identity_payload,
        "core_search_id": core_id,
        "content_fingerprint": fingerprint,
        "status": "SEALED_CORE_COMPLETE",
        "scope": {
            "development_only": True,
            "validation_or_final_rows_read": 0,
            "gamma_phase_authorized": True,
            "stage10_or_later_permitted": False,
            "portfolio_or_backtest_permitted": False,
        },
    }
    base = _safe_child_roots(
        database_root, policy, str(prepared.metadata["build_id"])
    )
    destination = base / "search_control" / "xgboost_core" / f"{core_id}.json"
    _create_safe_directory(destination.parent, trusted_root=database_root)
    expected_sha = _sha256_bytes(canonical_json_bytes(receipt))
    written_sha = atomic_write_json(destination, receipt, trusted_root=database_root)
    _require(written_sha == expected_sha, "Sealed XGBoost core receipt hash differs")
    _require(
        destination.is_file()
        and not destination.is_symlink()
        and sha256_file(destination) == expected_sha,
        "Sealed XGBoost core receipt failed reauthentication",
    )
    return SealedCoreReceipt(
        core_search_id=core_id,
        path=destination,
        receipt=receipt,
        sha256=expected_sha,
        gamma_parent_config_ids=parents,
    )


def _combined_artifacts(*values: PhaseArtifacts) -> PhaseArtifacts:
    predictions: dict[tuple[str, str, int | None], np.ndarray] = {}
    receipts: list[Mapping[str, Any]] = []
    hashes: dict[str, str] = {}
    for value in values:
        overlap = set(predictions).intersection(value.predictions)
        _require(not overlap, f"Prediction phases overlap: {sorted(overlap)[:1]}")
        predictions.update(value.predictions)
        receipts.extend(value.publication_fit_receipts)
        _require(
            not set(hashes).intersection(value.cache_sidecar_sha256s),
            "Cache sidecar job IDs overlap across phases",
        )
        hashes.update(value.cache_sidecar_sha256s)
    return PhaseArtifacts(predictions, tuple(receipts), dict(sorted(hashes.items())))


def execute_stage9_selector_search(
    *,
    prepared_root: Path,
    expected_build_id: str,
    expected_content_fingerprint: str,
    database_root: Path,
    policy_path: Path = DEFAULT_POLICY,
    progress: Progress | None = None,
    phase_executor: PhaseExecutor = execute_native_phase,
    heartbeat_seconds: float = _DEFAULT_HEARTBEAT_SECONDS,
    resume_authenticated_elastic_net_cache: bool = False,
    resume_authenticated_elastic_net_prefix: bool = False,
) -> Stage9SearchExecution:
    """Execute and seal the complete primary Stage 9 development search."""

    _require(
        not (
            resume_authenticated_elastic_net_cache
            and resume_authenticated_elastic_net_prefix
        ),
        "Elastic Net full-cache and partial-prefix resume modes are mutually exclusive",
    )
    emit = progress if progress is not None else (lambda _message: None)
    started = time.monotonic()
    policy_path = policy_path.expanduser().resolve()
    policy, _ = validate_contract(policy_path)
    validate_runtime_versions()
    database_root = Path(os.path.abspath(database_root.expanduser()))
    _require(
        database_root.is_dir()
        and not database_root.is_symlink()
        and str(database_root) == str(policy["database"]["root"]),
        "Search database root differs from the frozen policy",
    )
    emit("Stage 9 search: authenticating the completed nested prepared-data cache")
    prepared = load_prepared_selector_data(
        prepared_root,
        expected_build_id=expected_build_id,
        expected_content_fingerprint=expected_content_fingerprint,
    )
    _require(
        prepared.metadata["build_id"] == expected_build_id
        and prepared.metadata["content_fingerprint"] == expected_content_fingerprint,
        "Prepared selector data identity differs after loading",
    )
    domain = build_oof_domain(prepared.rows, prepared.folds)
    phase_results: list[NativePhaseResult] = []

    nn_configs = neural_configurations()
    nn_plan = build_native_phase_plan(
        prepared,
        family="neural_network",
        phase_id="neural_full_grid",
        configurations=nn_configs,
        database_root=database_root,
        policy_path=policy_path,
    )
    emit("Stage 9 search: starting 48-config neural-network OOF phase")
    nn_result = phase_executor(
        nn_plan, progress=emit, heartbeat_seconds=heartbeat_seconds
    )
    phase_results.append(nn_result)
    nn_artifacts = load_phase_artifacts(nn_plan, nn_result, prepared)
    nn_ids = tuple(value.config_id for value in nn_configs)
    nn_cube = _stochastic_cube(
        domain, prepared, nn_artifacts, config_ids=nn_ids
    )
    nn_evaluation = evaluate_family_predictions(
        domain.rows,
        model_family="neural_network",
        config_ids=nn_ids,
        predictions=nn_cube,
        seeds=DEVELOPMENT_SEEDS,
    )
    emit("Stage 9 search: neural OOF evaluation and shared bootstrap complete")

    core_configs = core_xgboost_configurations()
    core_plan = build_native_phase_plan(
        prepared,
        family="xgboost",
        phase_id="xgboost_core_grid",
        configurations=core_configs,
        database_root=database_root,
        policy_path=policy_path,
    )
    emit("Stage 9 search: starting 40-config XGBoost core OOF phase")
    core_result = phase_executor(
        core_plan, progress=emit, heartbeat_seconds=heartbeat_seconds
    )
    phase_results.append(core_result)
    core_artifacts = load_phase_artifacts(core_plan, core_result, prepared)
    core_ids = tuple(value.config_id for value in core_configs)
    core_cube = _stochastic_cube(
        domain, prepared, core_artifacts, config_ids=core_ids
    )
    core_evaluation = evaluate_family_predictions(
        domain.rows,
        model_family="xgboost",
        config_ids=core_ids,
        predictions=core_cube,
        seeds=DEVELOPMENT_SEEDS,
    )
    core_fit_frame = pd.DataFrame(core_artifacts.publication_fit_receipts)
    core_complexity = xgboost_complexity(
        core_fit_frame, configurations=core_configs
    )
    sealed_core = seal_xgboost_core_receipt(
        prepared=prepared,
        core_cube=core_cube,
        core_evaluation=core_evaluation,
        core_complexity=core_complexity,
        artifacts=core_artifacts,
        database_root=database_root,
        policy_path=policy_path,
    )
    emit(
        "Stage 9 search: XGBoost core sealed; gamma parents are "
        + ", ".join(sealed_core.gamma_parent_config_ids)
    )

    gamma_configs = derive_gamma_challengers(
        sealed_core.gamma_parent_config_ids,
        sealed_core_receipt_sha256=sealed_core.sha256,
    )
    gamma_authorization = {
        "sealed_core_receipt_sha256": sealed_core.sha256,
        "authorized_config_payload_sha256s": [
            configuration_payload_sha256(value) for value in gamma_configs
        ],
    }
    gamma_plan = build_native_phase_plan(
        prepared,
        family="xgboost",
        phase_id="xgboost_gamma_challengers",
        configurations=gamma_configs,
        database_root=database_root,
        policy_path=policy_path,
        gamma_authorization=gamma_authorization,
    )
    emit("Stage 9 search: starting eight sealed-parent XGBoost gamma challengers")
    gamma_result = phase_executor(
        gamma_plan, progress=emit, heartbeat_seconds=heartbeat_seconds
    )
    phase_results.append(gamma_result)
    gamma_artifacts = load_phase_artifacts(gamma_plan, gamma_result, prepared)
    gamma_ids = tuple(value.config_id for value in gamma_configs)
    gamma_cube = _stochastic_cube(
        domain, prepared, gamma_artifacts, config_ids=gamma_ids
    )
    xgb_configs = (*core_configs, *gamma_configs)
    xgb_ids = (*core_ids, *gamma_ids)
    xgb_cube = np.concatenate((core_cube, gamma_cube), axis=1)
    _require(
        xgb_cube.shape == (len(domain.rows), 48, 3),
        "Combined XGBoost OOF cube differs",
    )
    xgb_evaluation = evaluate_family_predictions(
        domain.rows,
        model_family="xgboost",
        config_ids=xgb_ids,
        predictions=xgb_cube,
        seeds=DEVELOPMENT_SEEDS,
    )
    emit("Stage 9 search: full XGBoost OOF evaluation complete")

    enet_plan = build_native_phase_plan(
        prepared,
        family="elastic_net",
        phase_id="elastic_net_full_path",
        configurations=(ELASTIC_NET_PATH_CONFIG_ID,),
        database_root=database_root,
        policy_path=policy_path,
    )
    emit("Stage 9 search: starting four sequential Elastic Net path jobs")
    if resume_authenticated_elastic_net_cache:
        enet_result = load_authenticated_cached_phase(
            enet_plan, prepared, progress=emit
        )
    elif resume_authenticated_elastic_net_prefix:
        _require(
            phase_executor is execute_native_phase,
            "Elastic Net partial-prefix resume requires the authoritative native executor",
        )
        enet_result = execute_native_phase(
            enet_plan,
            progress=emit,
            heartbeat_seconds=heartbeat_seconds,
            allow_authenticated_elastic_net_fold1_cache=True,
        )
    else:
        enet_result = phase_executor(
            enet_plan, progress=emit, heartbeat_seconds=heartbeat_seconds
        )
    phase_results.append(enet_result)
    enet_artifacts = load_phase_artifacts(enet_plan, enet_result, prepared)
    enet_ids = tuple(value.config_id for value in elastic_net_configurations())
    enet_cube = _elastic_cube(domain, prepared, enet_artifacts)
    enet_evaluation = evaluate_family_predictions(
        domain.rows,
        model_family="elastic_net",
        config_ids=enet_ids,
        predictions=enet_cube,
    )
    emit("Stage 9 search: Elastic Net OOF evaluation complete")

    xgb_artifacts = _combined_artifacts(core_artifacts, gamma_artifacts)
    all_artifacts = _combined_artifacts(
        nn_artifacts, xgb_artifacts, enet_artifacts
    )
    fit_receipts = pd.DataFrame(all_artifacts.publication_fit_receipts)
    counts = frozen_search_plan_counts()
    _require(
        len(fit_receipts) == counts.fit_receipts
        and fit_receipts["job_id"].nunique() == counts.fit_receipts,
        "Final normalized fit-receipt ledger differs",
    )
    complexities = {
        "neural_network": neural_complexity(fit_receipts, config_ids=nn_ids),
        "xgboost": xgboost_complexity(
            fit_receipts, configurations=xgb_configs
        ),
        "elastic_net": elastic_net_complexity(
            fit_receipts, config_ids=enet_ids
        ),
    }
    evaluations = {
        "neural_network": nn_evaluation,
        "xgboost": xgb_evaluation,
        "elastic_net": enet_evaluation,
    }
    raw_predictions = {
        "neural_network": nn_cube,
        "xgboost": xgb_cube,
        "elastic_net": enet_cube,
    }
    emit("Stage 9 search: all 346 development configurations evaluated; sealing search bundle")
    publication = publish_completed_search_bundle(
        CompletedSearchInputs(
            prepared_metadata=prepared.metadata,
            domain=domain,
            raw_predictions=raw_predictions,
            evaluations=evaluations,
            fit_receipts=fit_receipts,
            complexities=complexities,
            shortlist=None,
            sealed_xgboost_core_receipt_sha256=sealed_core.sha256,
            gamma_parent_config_ids=sealed_core.gamma_parent_config_ids,
            extra_code_paths=(
                "src/portfolio_bot/selectors/search_execution.py",
                "scripts/run_stage_09_selector_search.py",
            ),
        ),
        database_root=database_root,
        policy_path=policy_path,
        progress=emit,
    )
    total_hits = sum(value.cache_hits for value in phase_results)
    elapsed = float(time.monotonic() - started)
    emit(
        f"Stage 9 search bundle complete: {publication.search_id}; "
        f"{total_hits}/{sum(value.job_count for value in phase_results)} fit jobs resumed"
    )
    return Stage9SearchExecution(
        publication=publication,
        sealed_core=sealed_core,
        phase_results=tuple(phase_results),
        evaluated_configurations=counts.evaluated_configurations,
        fit_receipts=len(fit_receipts),
        cache_hits=total_hits,
        elapsed_seconds=elapsed,
    )


__all__ = [
    "NativePhasePlan",
    "NativePhaseResult",
    "PhaseArtifacts",
    "PhaseExecutor",
    "SealedCoreReceipt",
    "Stage9SearchExecution",
    "Stage9SearchExecutionError",
    "build_native_phase_plan",
    "execute_native_phase",
    "execute_stage9_selector_search",
    "load_authenticated_cached_phase",
    "load_phase_artifacts",
    "seal_xgboost_core_receipt",
]
