"""Native-free bootstrap and job identity for family-isolated Stage 9 workers."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from portfolio_bot.selectors.contract import (
    REQUIRED_RUNTIME_VERSIONS,
    canonical_json_bytes,
    validate_contract,
)
from portfolio_bot.selectors.model_specs import (
    DEVELOPMENT_SEEDS,
    ELASTIC_NET_PATH_CONFIG_ID,
    canonical_xgboost_configuration,
    configuration_payload_sha256,
    elastic_net_path_payload_sha256,
    neural_configurations,
)
from portfolio_bot.selectors.native_process import (
    NATIVE_FAMILIES,
    configure_single_thread_environment,
    require_spawned_native_worker,
)


_FAMILY_MODULE = {
    "neural_network": "portfolio_bot.selectors.neural_model",
    "xgboost": "portfolio_bot.selectors.xgboost_model",
    "elastic_net": "portfolio_bot.selectors.elastic_net",
}
_SHA256 = re.compile(r"[0-9a-f]{64}")
_FOLDS = frozenset({"dev_fold_1", "dev_fold_2", "dev_fold_3", "dev_fold_4"})
RUNTIME_VERSIONS_SHA256 = hashlib.sha256(
    canonical_json_bytes(dict(REQUIRED_RUNTIME_VERSIONS))
).hexdigest()


@dataclass(frozen=True)
class NativeFitSpec:
    family: str
    config_id: str
    fold_id: str
    seed: int | None
    config_payload_sha256: str
    runtime_versions_sha256: str
    rules_sha256: str
    input_bundle_sha256: str
    code_sha256: str
    support_sha256: str
    prepared_root: str
    cache_path: str
    sealed_core_receipt_sha256: str | None = None

    def validated(self) -> "NativeFitSpec":
        if self.family not in NATIVE_FAMILIES:
            raise ValueError("NativeFitSpec family is not frozen")
        if not self.config_id or self.fold_id not in _FOLDS:
            raise ValueError("NativeFitSpec config or fold is not frozen")
        if self.family == "neural_network":
            canonical = {value.config_id: value for value in neural_configurations()}.get(
                self.config_id
            )
            expected_config_sha = (
                None if canonical is None else configuration_payload_sha256(canonical)
            )
        elif self.family == "xgboost":
            try:
                canonical = canonical_xgboost_configuration(self.config_id)
            except ValueError:
                canonical = None
            expected_config_sha = (
                None if canonical is None else configuration_payload_sha256(canonical)
            )
        else:
            expected_config_sha = (
                elastic_net_path_payload_sha256()
                if self.config_id == ELASTIC_NET_PATH_CONFIG_ID
                else None
            )
        if expected_config_sha is None or self.config_payload_sha256 != expected_config_sha:
            raise ValueError("NativeFitSpec config ID/payload hash is not canonical")
        if self.family in {"neural_network", "xgboost"}:
            if self.seed not in DEVELOPMENT_SEEDS:
                raise ValueError("Stochastic NativeFitSpec seed is outside {11, 23, 47}")
        elif self.seed is not None:
            raise ValueError("Elastic Net NativeFitSpec must not carry a seed")
        for value in (
            self.rules_sha256,
            self.config_payload_sha256,
            self.runtime_versions_sha256,
            self.input_bundle_sha256,
            self.code_sha256,
            self.support_sha256,
        ):
            if _SHA256.fullmatch(value) is None:
                raise ValueError("NativeFitSpec contains an invalid SHA-256")
        if self.runtime_versions_sha256 != RUNTIME_VERSIONS_SHA256:
            raise ValueError("NativeFitSpec runtime binding differs from the frozen versions")
        is_gamma = self.family == "xgboost" and self.config_id.endswith(("_g01", "_g10"))
        if is_gamma:
            if (
                self.sealed_core_receipt_sha256 is None
                or _SHA256.fullmatch(self.sealed_core_receipt_sha256) is None
            ):
                raise ValueError("Gamma NativeFitSpec lacks a sealed core receipt binding")
        elif self.sealed_core_receipt_sha256 is not None:
            raise ValueError("A non-gamma NativeFitSpec cannot carry a core receipt binding")
        policy, _ = validate_contract()
        if self.rules_sha256 != policy["physical_rules"]["sha256"]:
            raise ValueError("NativeFitSpec rules hash differs from the current frozen contract")
        for value in (self.prepared_root, self.cache_path):
            path = Path(value)
            if not path.is_absolute() or ".." in path.parts:
                raise ValueError("NativeFitSpec paths must be absolute and traversal-free")
        return self

    def job_identity(self) -> dict[str, Any]:
        self.validated()
        value = asdict(self)
        value.pop("prepared_root")
        value.pop("cache_path")
        return value

    def expected_array_schema(self, outer_validation_rows: int) -> dict[str, dict[str, Any]]:
        if outer_validation_rows <= 0:
            raise ValueError("NativeFitSpec outer-validation row count must be positive")
        shape = (
            [int(outer_validation_rows), 250]
            if self.family == "elastic_net"
            else [int(outer_validation_rows)]
        )
        return {"predictions": {"dtype": "<f4", "shape": shape}}


def load_native_family_module(family: str) -> ModuleType:
    """Lazy-import exactly one native family inside a multiprocessing spawn child."""

    require_spawned_native_worker(family)
    configure_single_thread_environment()
    return importlib.import_module(_FAMILY_MODULE[family])


def run_native_family_shard(
    *,
    family: str,
    handler_module: str,
    handler_factory: str,
    initialization_payload: dict[str, Any],
    request_queue: Any,
    result_queue: Any,
) -> None:
    """Serve many small fit specs from one long-lived family/fold spawn child.

    The native-free handler factory is called once.  It is responsible for
    authenticating and memory-mapping its prepared fold data at initialization,
    then returning a callable that consumes :class:`NativeFitSpec` objects and
    writes prediction artifacts through the authenticated cache API.  Only
    receipt-sized JSON objects may cross the result queue.
    """

    native_module = load_native_family_module(family)
    gamma_authorization = initialization_payload.get("gamma_authorization")
    if gamma_authorization is not None:
        if family != "xgboost" or not isinstance(gamma_authorization, dict):
            raise ValueError("Only an XGBoost shard may carry gamma authorization")
        receipt_sha = gamma_authorization.get("sealed_core_receipt_sha256")
        authorized_hashes = tuple(
            gamma_authorization.get("authorized_config_payload_sha256s", ())
        )
        if (
            not isinstance(receipt_sha, str)
            or _SHA256.fullmatch(receipt_sha) is None
            or len(authorized_hashes) != 8
            or len(set(authorized_hashes)) != 8
            or any(_SHA256.fullmatch(str(value)) is None for value in authorized_hashes)
        ):
            raise ValueError("XGBoost gamma shard authorization is not a sealed eight-config set")
    factory_module = importlib.import_module(handler_module)
    factory = getattr(factory_module, handler_factory, None)
    if not callable(factory):
        raise RuntimeError("Stage 9 shard handler factory is absent or not callable")
    handler = factory(
        native_module=native_module,
        initialization_payload=dict(initialization_payload),
    )
    if not callable(handler):
        raise RuntimeError("Stage 9 shard handler factory did not return a callable")
    result_queue.put({"status": "READY", "family": family, "pid": os.getpid()})
    try:
        while True:
            request = request_queue.get()
            if request == {"command": "STOP"}:
                result_queue.put({"status": "STOPPED", "family": family})
                break
            try:
                specification = (
                    request
                    if isinstance(request, NativeFitSpec)
                    else NativeFitSpec(**dict(request))
                ).validated()
                if specification.family != family:
                    raise ValueError("Stage 9 shard received an opposing-family job")
                if specification.sealed_core_receipt_sha256 is not None:
                    authorization = initialization_payload.get("gamma_authorization")
                    if not isinstance(authorization, dict):
                        raise ValueError("Gamma shard was not initialized with authorization")
                    if (
                        authorization.get("sealed_core_receipt_sha256")
                        != specification.sealed_core_receipt_sha256
                        or specification.config_payload_sha256
                        not in set(authorization.get("authorized_config_payload_sha256s", ()))
                    ):
                        raise ValueError("Gamma job is absent from the authenticated shard authorization")
                receipt = handler(specification)
                if not isinstance(receipt, dict):
                    raise TypeError("Stage 9 shard handler must return a receipt dictionary")
                # This rejects ndarray/bytes payloads and prevents large model
                # outputs from accidentally crossing the process queue.
                json.dumps(receipt, sort_keys=True, allow_nan=False)
                result_queue.put(
                    {
                        "status": "COMPLETE",
                        "job_identity": specification.job_identity(),
                        "receipt": receipt,
                    }
                )
            except Exception as exc:
                result_queue.put(
                    {
                        "status": "ERROR",
                        "family": family,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
    finally:
        close = getattr(handler, "close", None)
        if callable(close):
            close()


__all__ = [
    "NativeFitSpec",
    "RUNTIME_VERSIONS_SHA256",
    "load_native_family_module",
    "run_native_family_shard",
]
