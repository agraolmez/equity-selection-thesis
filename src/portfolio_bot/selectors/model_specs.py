"""Native-runtime-free canonical Stage 9 selector specifications."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, replace
from itertools import product
from typing import Iterable


DEVELOPMENT_SEEDS = (11, 23, 47)

NEURAL_HIDDEN_LAYER_GRID = (
    (16,),
    (32,),
    (64,),
    (32, 16),
    (64, 32),
    (32, 16, 8),
)
NEURAL_LEARNING_RATE_GRID = (3e-4, 1e-3)
NEURAL_L1_GRID = (0.0, 1e-4)
NEURAL_DROPOUT_GRID = (0.0, 0.10)

XGBOOST_MAX_DEPTHS = (1, 2, 3, 4)
XGBOOST_CORE_ROWS = (
    (0.0050, 0.02000, 1.00, 0.40, 300.0, 0.0),
    (0.0075, 0.00050, 0.55, 0.87, 0.3, 0.001),
    (0.0100, 0.01000, 0.85, 0.60, 30.0, 0.0),
    (0.0150, 0.00100, 0.65, 1.00, 3.0, 0.01),
    (0.0200, 0.00500, 0.95, 0.47, 100.0, 0.0),
    (0.0300, 0.00075, 0.50, 0.73, 1.0, 0.10),
    (0.0400, 0.01500, 0.75, 0.93, 10.0, 0.0),
    (0.0600, 0.00200, 0.90, 0.53, 0.6, 1.0),
    (0.0800, 0.00750, 0.60, 0.80, 60.0, 0.0),
    (0.1000, 0.00350, 0.80, 0.67, 6.0, 10.0),
)
ELASTIC_NET_RHO_GRID = (0.0, 0.25, 0.50, 0.75, 1.0)
ELASTIC_NET_LAMBDA_POINTS = 50
ELASTIC_NET_PATH_CONFIG_ID = "enet_all_250"
_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class NeuralConfig:
    config_id: str
    hidden_layers: tuple[int, ...]
    learning_rate: float
    l1: float
    dropout: float


@dataclass(frozen=True)
class XGBoostConfig:
    config_id: str
    core_config_id: str
    row: int
    max_depth: int
    eta: float
    child_share: float
    subsample: float
    colsample_bytree: float
    reg_lambda: float
    reg_alpha: float
    gamma: float = 0.0
    challenger: bool = False


@dataclass(frozen=True)
class ElasticNetConfig:
    config_id: str
    rho: float
    lambda_index: int
    lambda_fraction: float


def neural_configurations() -> tuple[NeuralConfig, ...]:
    configs = tuple(
        NeuralConfig(
            config_id=f"nn_{ordinal:03d}",
            hidden_layers=tuple(layers),
            learning_rate=float(learning_rate),
            l1=float(l1),
            dropout=float(dropout),
        )
        for ordinal, (layers, learning_rate, l1, dropout) in enumerate(
            product(
                NEURAL_HIDDEN_LAYER_GRID,
                NEURAL_LEARNING_RATE_GRID,
                NEURAL_L1_GRID,
                NEURAL_DROPOUT_GRID,
            ),
            start=1,
        )
    )
    if len(configs) != 48:
        raise AssertionError("The frozen NN grid must contain exactly 48 configs")
    return configs


def core_xgboost_configurations() -> tuple[XGBoostConfig, ...]:
    values: list[XGBoostConfig] = []
    ordinal = 0
    for row, params in enumerate(XGBOOST_CORE_ROWS, start=1):
        eta, child_share, subsample, column_sample, reg_lambda, reg_alpha = params
        for depth in XGBOOST_MAX_DEPTHS:
            ordinal += 1
            config_id = f"xgb_core_{ordinal:03d}"
            values.append(
                XGBoostConfig(
                    config_id=config_id,
                    core_config_id=config_id,
                    row=row,
                    max_depth=depth,
                    eta=eta,
                    child_share=child_share,
                    subsample=subsample,
                    colsample_bytree=column_sample,
                    reg_lambda=reg_lambda,
                    reg_alpha=reg_alpha,
                )
            )
    if len(values) != 40:
        raise AssertionError("The frozen XGBoost core grid must contain 40 configs")
    return tuple(values)


def elastic_net_configurations() -> tuple[ElasticNetConfig, ...]:
    result: list[ElasticNetConfig] = []
    for rho_index, rho in enumerate(ELASTIC_NET_RHO_GRID):
        span = 6.0 if rho == 0.0 else 4.0
        for lambda_index in range(1, ELASTIC_NET_LAMBDA_POINTS + 1):
            exponent = -span * float(lambda_index - 1) / float(
                ELASTIC_NET_LAMBDA_POINTS - 1
            )
            result.append(
                ElasticNetConfig(
                    config_id=f"enet_r{rho_index:03d}_l{lambda_index:03d}",
                    rho=rho,
                    lambda_index=lambda_index,
                    lambda_fraction=float(10.0**exponent),
                )
            )
    if len(result) != 250:
        raise AssertionError("The frozen Elastic Net grid must contain 250 configs")
    return tuple(result)


def elastic_net_lambda_value(config: ElasticNetConfig, lambda_ref: float) -> float:
    if not math.isfinite(float(lambda_ref)) or float(lambda_ref) < 0.0:
        raise ValueError("lambda_ref must be finite and nonnegative")
    if config.rho == 0.0:
        return float(100.0 * float(lambda_ref) * config.lambda_fraction)
    return float(float(lambda_ref) / config.rho * config.lambda_fraction)


def configuration_payload_sha256(config: object) -> str:
    if not isinstance(config, (NeuralConfig, XGBoostConfig, ElasticNetConfig)):
        raise TypeError("Unsupported Stage 9 configuration payload")
    payload = (json.dumps(asdict(config), sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def elastic_net_path_payload_sha256() -> str:
    payload = (
        json.dumps(
            [asdict(value) for value in elastic_net_configurations()],
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def canonical_xgboost_configuration(config_id: str) -> XGBoostConfig:
    core = {value.config_id: value for value in core_xgboost_configurations()}
    if config_id in core:
        return core[config_id]
    match = re.fullmatch(r"(xgb_core_\d{3})_(g01|g10)", config_id)
    if match is None or match.group(1) not in core:
        raise ValueError("XGBoost config ID is outside the canonical core/gamma structure")
    base = core[match.group(1)]
    gamma = 0.1 if match.group(2) == "g01" else 1.0
    return replace(base, config_id=config_id, gamma=gamma, challenger=True)


def require_canonical_neural_config(config: NeuralConfig) -> NeuralConfig:
    canonical = {value.config_id: value for value in neural_configurations()}.get(config.config_id)
    if canonical is None or config != canonical:
        raise ValueError("Authoritative Stage 9 NN config is outside the frozen grid")
    return canonical


def require_canonical_xgboost_core_config(config: XGBoostConfig) -> XGBoostConfig:
    canonical = {
        value.config_id: value for value in core_xgboost_configurations()
    }.get(config.config_id)
    if canonical is None or config != canonical:
        raise ValueError("Authoritative Stage 9 XGBoost config is outside the frozen core grid")
    return canonical


def require_canonical_xgboost_config(config: XGBoostConfig) -> XGBoostConfig:
    canonical = canonical_xgboost_configuration(config.config_id)
    if config != canonical:
        raise ValueError("Authoritative Stage 9 XGBoost config payload is not canonical")
    return canonical


def require_canonical_elastic_net_config(config: ElasticNetConfig) -> ElasticNetConfig:
    canonical = {
        value.config_id: value for value in elastic_net_configurations()
    }.get(config.config_id)
    if canonical is None or config != canonical:
        raise ValueError("Authoritative Stage 9 Elastic Net config is outside the frozen grid")
    return canonical


def gamma_challenger_configurations(
    strongest_core: Iterable[XGBoostConfig],
    *,
    authenticated_core_receipt_sha256: str | None = None,
) -> tuple[XGBoostConfig, ...]:
    """Derive challengers only from four externally authenticated core winners."""

    selected = tuple(strongest_core)
    if (
        authenticated_core_receipt_sha256 is None
        or _SHA256.fullmatch(authenticated_core_receipt_sha256) is None
    ):
        raise RuntimeError(
            "Gamma challengers require an authenticated sealed core-search receipt"
        )
    if len(selected) != 4 or len({value.config_id for value in selected}) != 4:
        raise ValueError("Exactly four distinct authenticated core configurations are required")
    canonical = tuple(require_canonical_xgboost_core_config(value) for value in selected)
    result: list[XGBoostConfig] = []
    for base in canonical:
        result.extend(
            (
                replace(
                    base,
                    config_id=f"{base.config_id}_g01",
                    gamma=0.1,
                    challenger=True,
                ),
                replace(
                    base,
                    config_id=f"{base.config_id}_g10",
                    gamma=1.0,
                    challenger=True,
                ),
            )
        )
    return tuple(result)


__all__ = [
    "DEVELOPMENT_SEEDS",
    "ELASTIC_NET_RHO_GRID",
    "ELASTIC_NET_PATH_CONFIG_ID",
    "ElasticNetConfig",
    "NeuralConfig",
    "XGBoostConfig",
    "core_xgboost_configurations",
    "canonical_xgboost_configuration",
    "configuration_payload_sha256",
    "elastic_net_configurations",
    "elastic_net_lambda_value",
    "elastic_net_path_payload_sha256",
    "gamma_challenger_configurations",
    "neural_configurations",
    "require_canonical_neural_config",
    "require_canonical_elastic_net_config",
    "require_canonical_xgboost_core_config",
    "require_canonical_xgboost_config",
]
