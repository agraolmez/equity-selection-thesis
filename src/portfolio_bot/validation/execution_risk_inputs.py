"""Exact verified risk/cost registrations for empirical Stage 11 execution."""

from __future__ import annotations

import datetime as dt
import hashlib
import os
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from portfolio_bot.risk_cost.contract import published_schemas
from portfolio_bot.validation.access import PayloadRole
from portfolio_bot.validation.execution_access import (
    ExecutionDateAxis,
    RegisteredExecutionPayload,
)
from portfolio_bot.validation.predictor_adapter import (
    AuthenticatedValidationPredictor,
)


FIRST_SIGNAL_DATE = dt.date(2014, 12, 31)
LAST_SIGNAL_DATE = dt.date(2019, 10, 31)
RISK_COST_REGISTRATION_NAMES = (
    "validation_risk_cost_audit",
    "validation_risk_cost_inputs",
    "validation_factor_covariance",
    "validation_risk_cost_month_summary",
)
_SCHEMA_NAME = {
    "validation_risk_cost_audit": "development_risk_cost_audit.parquet",
    "validation_risk_cost_inputs": "development_risk_cost_inputs.parquet",
    "validation_factor_covariance": "development_factor_covariance.parquet",
    "validation_risk_cost_month_summary": (
        "development_risk_cost_month_summary.parquet"
    ),
}
_ROLE = {
    "validation_risk_cost_audit": PayloadRole.VALIDATION_RISK_INPUT,
    "validation_risk_cost_inputs": PayloadRole.VALIDATION_COST_INPUT,
    "validation_factor_covariance": PayloadRole.VALIDATION_RISK_INPUT,
    "validation_risk_cost_month_summary": PayloadRole.VALIDATION_RISK_INPUT,
}


class ExecutionRiskInputError(RuntimeError):
    """Raised when verified risk/cost payload authority differs."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ExecutionRiskInputError(message)


def _schema_sha256(name: str) -> str:
    schema = published_schemas()[_SCHEMA_NAME[name]]
    return hashlib.sha256(schema.serialize().to_pybytes()).hexdigest()


def build_authenticated_risk_cost_registrations(
    predictor: AuthenticatedValidationPredictor,
) -> Mapping[str, RegisteredExecutionPayload]:
    """Derive four broker records only from the verified 12-role receipt set."""

    _require(
        type(predictor) is AuthenticatedValidationPredictor,
        "risk/cost registration requires the authenticated predictor authority",
    )
    verification = predictor.validation_data_verification
    schemas = published_schemas()
    root = verification.candidate_root
    _require(root.is_absolute(), "verified candidate root is not absolute")
    records: dict[str, RegisteredExecutionPayload] = {}
    for name in RISK_COST_REGISTRATION_NAMES:
        receipt = verification.payload_receipt(name)
        filename = str(receipt["path"])
        path = Path(os.path.abspath(root / filename))
        _require(
            path.parent == root
            and path.name == filename
            and str(receipt["schema_sha256"]) == _schema_sha256(name)
            and receipt["minimum_signal_date"] == FIRST_SIGNAL_DATE.isoformat()
            and receipt["maximum_signal_date"] == LAST_SIGNAL_DATE.isoformat(),
            f"verified risk/cost receipt scope or schema differs: {name}",
        )
        schema = schemas[_SCHEMA_NAME[name]]
        records[name] = RegisteredExecutionPayload(
            role=_ROLE[name],
            path=path,
            sha256=str(receipt["sha256"]),
            size_bytes=int(receipt["size_bytes"]),
            allowed_columns=tuple(schema.names),
            date_column="signal_date",
            date_axis=ExecutionDateAxis.SIGNAL,
            minimum_date=FIRST_SIGNAL_DATE,
            maximum_date=LAST_SIGNAL_DATE,
            holding_month_column=None,
            trusted_schema_sha256=None,
        )
    validate_risk_cost_registrations(records)
    return MappingProxyType(records)


def validate_risk_cost_registrations(
    records: Mapping[str, RegisteredExecutionPayload],
) -> None:
    """Reject omitted, substituted, widened, or relabelled risk/cost records."""

    _require(
        set(records) == set(RISK_COST_REGISTRATION_NAMES),
        "execution risk/cost registration set differs",
    )
    schemas = published_schemas()
    for name in RISK_COST_REGISTRATION_NAMES:
        record = records[name]
        _require(
            type(record) is RegisteredExecutionPayload
            and record.role is _ROLE[name]
            and record.allowed_columns == tuple(schemas[_SCHEMA_NAME[name]].names)
            and record.date_column == "signal_date"
            and record.date_axis is ExecutionDateAxis.SIGNAL
            and record.minimum_date == FIRST_SIGNAL_DATE
            and record.maximum_date == LAST_SIGNAL_DATE
            and record.holding_month_column is None
            and record.trusted_schema_sha256 is None,
            f"execution risk/cost registration differs: {name}",
        )


__all__ = [
    "ExecutionRiskInputError",
    "FIRST_SIGNAL_DATE",
    "LAST_SIGNAL_DATE",
    "RISK_COST_REGISTRATION_NAMES",
    "build_authenticated_risk_cost_registrations",
    "validate_risk_cost_registrations",
]
