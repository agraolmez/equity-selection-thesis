"""Process-local guards for native Stage 9 model runtimes.

Authoritative selector fits use separate ``spawn`` process families.  This
module deliberately imports no NumPy, sklearn, XGBoost, or PyTorch code so an
orchestrator can import it safely before creating a family-specific worker.
"""

from __future__ import annotations

import multiprocessing as mp
import os


NATIVE_FAMILIES = frozenset({"elastic_net", "xgboost", "neural_network"})
_CLAIMED_FAMILY: str | None = None
_THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def configure_single_thread_environment() -> None:
    """Set native thread limits before a family runtime is lazily imported."""

    for name in _THREAD_ENVIRONMENT:
        os.environ[name] = "1"


def claim_native_family(family: str) -> None:
    """Prevent two authoritative native model families sharing one process."""

    global _CLAIMED_FAMILY
    if family not in NATIVE_FAMILIES:
        raise RuntimeError(f"Unknown Stage 9 native model family: {family}")
    if _CLAIMED_FAMILY is not None and _CLAIMED_FAMILY != family:
        raise RuntimeError(
            "Stage 9 native process isolation violation: "
            f"process already claimed by {_CLAIMED_FAMILY}, cannot import {family}"
        )
    _CLAIMED_FAMILY = family


def require_spawned_native_worker(family: str) -> None:
    """Reject authoritative fits outside a family-specific spawn child."""

    claim_native_family(family)
    start_method = mp.get_start_method(allow_none=True)
    if (
        mp.current_process().name == "MainProcess"
        or mp.parent_process() is None
        or start_method != "spawn"
    ):
        raise RuntimeError(
            "Authoritative Stage 9 native fits require a separate multiprocessing "
            f"spawn worker for family {family}"
        )


__all__ = [
    "NATIVE_FAMILIES",
    "claim_native_family",
    "configure_single_thread_environment",
    "require_spawned_native_worker",
]
