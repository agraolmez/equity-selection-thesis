"""Authorization-first Stage 13 production orchestration contracts.

The public Stage 13 entry point has no injectable paths, readers, calendars, or
callbacks.  This module contains the resumable production runner used after the
fixed Stage 12 binding, separate Stage 13 approval, and verified final-input
registration pointer have all authenticated.

The small callable seam in this module is private and test-only.  It exercises
the checkpoint and cross-segment state machine without granting the public
entry point an override surface.  The native empirical adapter authenticates
each frozen registration, commits decisions before opening accounting or
outcome rows, and persists content-addressed state after every completed month.
"""

from __future__ import annotations

import datetime as dt
import base64
import hashlib
import json
import math
import multiprocessing as mp
import os
import re
import shutil
import stat
import tempfile
import traceback
from dataclasses import dataclass, replace
from pathlib import Path
from queue import Empty
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.ipc as pa_ipc
import pyarrow.parquet as pq

from portfolio_bot.final_test.access import FinalAccessBroker
from portfolio_bot.final_test.access import FinalPayloadRole
from portfolio_bot.final_test.authorization import (
    Stage12VerifiedBinding,
    Stage13ApprovalReceipt,
    Stage13AuthorizationError,
    require_stage13_unblind_authorization,
)
from portfolio_bot.final_test.calendar import (
    FinalPeriodSegment,
    final_holding_month_descriptor,
    shift_month,
)
from portfolio_bot.final_test.checkpoint import (
    CheckpointPlan,
    LoadedCheckpoint,
    MonthlyCheckpointRecord,
    MonthlyCheckpointStore,
)
from portfolio_bot.final_test.economic_kernel import (
    ACTIVE_STRATEGY_IDS,
    CALIBRATED_STRATEGY_IDS,
    CALIBRATION_STATE_KEYS,
    ELASTIC_NET_STRATEGY_ID,
    EQUAL_WEIGHT_STRATEGY_ID,
    FROZEN_FINAL_ECONOMIC_KERNEL_CONTRACT,
    MODEL_STATE_KEYS,
    PRIMARY_REFIT_SIGNAL_MONTHS,
    PRIMARY_STRATEGY_ID,
    RANK_STRATEGY_ID,
    REFERENCE_NAV_DOLLARS,
    TRAILING_RESIDUAL_STRATEGY_ID,
    VQM_STRATEGY_ID,
    SHARED_CALIBRATION_POLICY_ID,
    SHARED_CALIBRATION_SOURCE_SCOPE,
    FeatureScope,
    FinalCalibrationObservation,
    FinalCalibrationReceipt,
    FinalForecastReceipt,
    FinalKernelState,
    PortfolioSnapshot,
    SeedForecastBlock,
    StrategyMonthDecisionInput,
    StrategyMonthResult,
    account_certified_strategy_month_decision,
    aggregate_selected_forecasts,
    build_stage13_kernel_document_rows,
    fit_final_calibration,
    is_quarterly_refit,
    prepare_certified_strategy_month_decision,
    validate_kernel_state_successor,
)
from portfolio_bot.final_test.benchmarks import (
    VQM_FEATURES,
    trailing_residual_mean_estimate,
    value_quality_momentum_scores,
)
from portfolio_bot.final_test.orchestration import (
    FROZEN_SEGMENT_PLANS,
    FROZEN_EXACT_COLUMNS,
    PROJECT_ROOT,
    STAGE13_APPROVAL_RECEIPT,
    STAGE13_CHECKPOINT_ROOT,
    STAGE13_STATE_FORMAT,
    FrozenMonthRequest,
    FrozenSegmentPlan,
    SegmentExecutionResult,
    Stage13ExecutionError,
    Stage13RunResult,
    Stage13Segment,
    frozen_execution_config_sha256,
)
from portfolio_bot.final_test.results import (
    ACCESS_LEDGER_FILE,
    ALLOCATION_EVIDENCE_FILE,
    COST_CONSTRAINT_MATERIALITY_FILE,
    DAILY_ACCOUNTING_FILE,
    FACTOR_BENCHMARK_FILE,
    MODEL_EVIDENCE_FILE,
    MONTHLY_ACCOUNTING_FILE,
    STAGE13_RESULT_FILES,
    DESCRIPTIVE_BENCHMARK_IDS,
    Stage13ResultSet,
    Stage13ResultError,
    canonical_json_bytes as result_json_bytes,
)
from portfolio_bot.final_test.production_results import (
    EmpiricalAuxiliaryEvidence,
    EmpiricalMonthResultFragment,
    assemble_empirical_stage13_results,
)
from portfolio_bot.selectors.inputs import (
    FEATURE_SPINE_COLUMNS,
    Stage9Inputs,
    resolve_stage9_inputs,
)
from portfolio_bot.validation.access import PayloadRole
from portfolio_bot.validation.execution_input_manifest import (
    AuthenticatedDevelopmentOOF,
    FROZEN_EXECUTION_INPUT_MANIFEST_PATH,
    FROZEN_EXECUTION_INPUT_MANIFEST_SHA256,
    _authenticate_development_oof,
)
from portfolio_bot.validation.empirical_selector_backend import (
    PreparedRefit,
    _model_code_sha256,
    prepare_authenticated_refit,
    prepare_authenticated_score,
)
from portfolio_bot.validation.locked_selection import (
    LOCKED_STAGE9_SHORTLIST_SHA256,
)
from portfolio_bot.validation.predictor_adapter import (
    DATABASE_ROOT as STAGE11_DATABASE_ROOT,
    AuthenticatedValidationPredictor,
    authenticate_validation_predictor_manifest,
)
from portfolio_bot.selectors.native_worker import RUNTIME_VERSIONS_SHA256
from portfolio_bot.selectors.metrics import (
    centered_average_rank,
    correlation_with_status,
)
from portfolio_bot.validation.selector_runtime import (
    LoadedSelectorCheckpoint,
    NativeModelArtifact,
    PersistedPreprocessorState,
    ValidationRefitIdentity,
    build_refit_identity,
    load_array_for_spawn,
    persist_array_for_spawn,
    persist_validation_refit_arrays_for_spawn,
)
from portfolio_bot.risk_cost.contract import FACTOR_NAMES, PRESENT
from portfolio_bot.risk_cost.risk_math import type7_quantile
from portfolio_bot.validation.accounting import PortfolioState
from portfolio_bot.validation.corporate_actions import (
    CorporateAction,
    CorporateActionLedger,
)
from portfolio_bot.validation.daily_engine import MarkContinuityLedger
from portfolio_bot.validation.problem_builder import (
    AuthenticatedMonthlyIdentity,
    InheritedHoldings,
    MarketSnapshot,
    MonthlyRiskInputs,
    ScoredSnapshot,
)
from portfolio_bot.validation.session_authority import (
    authenticate_final_calendar_row,
    authenticate_final_holding_period_session_authority,
)
from portfolio_bot.final_test.input_materialization import (
    project_final_daily_accounting_inputs,
)


PRODUCTION_RUNTIME_ID = "stage13-authenticated-production-runtime-v2"
_NATIVE_ADAPTER_STATE_SCHEMA_VERSION = 2
PREFINAL_REFIT_INPUT_STATUS = (
    "AUTHENTICATED_STAGE9_STAGE11_PREFINAL_REFIT_HISTORY_AVAILABLE"
)
KERNEL_DETAIL_STATUS = "CERTIFIED_DECISION_ACCOUNTING_AND_PUBLICATION_DETAIL_AVAILABLE"
NATIVE_MONTH_ADAPTER_STATUS = "AUTHENTICATED_RESUMABLE_NATIVE_MONTH_ADAPTER_AVAILABLE"
PRODUCTION_STATE_SCHEMA_VERSION = 2

STAGE11_EXECUTION_VERIFIED_POINTER = (
    PROJECT_ROOT
    / "artifacts/phase_05_validation/LATEST_STAGE_11_EXECUTION_VERIFIED.json"
)
STAGE11_EXECUTION_VERIFIED_POINTER_SHA256 = (
    "abb9153df655f6ace66c6bd2729f065c6bb381ddf925f19e6d7bee4f061c2232"
)
STAGE11_PRIMARY_BUILD_ID = "stage11-validation-8d5e7f863ee093089520457e"
STAGE11_PRIMARY_MANIFEST_SHA256 = (
    "28481a09d5f933ed97f54b7441b45e9ac728ac4616e40fb76ce2b8f5abf60801"
)
STAGE11_VERIFICATION_ID = "verify-86a1e2d42b4c84f0"
STAGE11_OUTCOME_MEMBER = "validation_residual_outcomes.parquet"
STAGE11_SELECTED_SCORE_MEMBER = "selected_scores.parquet"
STAGE11_PRIMARY_MEMBERS = frozenset(
    {
        "raw_forecasts.parquet",
        STAGE11_OUTCOME_MEMBER,
        "family_monthly_ic.parquet",
        "ensemble_monthly_ic.parquet",
        "selected_scores.parquet",
        "calibration_schedule.parquet",
        "execution_sessions.parquet",
        "portfolio_weights.parquet",
        "portfolio_trades.parquet",
        "daily_accounting.parquet",
        "monthly_metrics.parquet",
        "execution_plan.json",
        "selection.json",
        "access_ledger.json",
        "manifest.json",
        "_SUCCESS.json",
    }
)

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_TEST_ADAPTER_TOKEN = object()


class Stage13ProductionRuntimeError(Stage13ExecutionError):
    """Raised when authenticated Stage 13 execution cannot proceed safely."""


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise Stage13ProductionRuntimeError(
            "production checkpoint state is not finite canonical JSON"
        ) from exc


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: object) -> str:
    return _sha256_bytes(_canonical_json_bytes(value))


def _require_hash(value: object, *, field: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise Stage13ProductionRuntimeError(f"{field} must be an exact SHA-256")
    return value


def _strict_hash_mapping(
    value: Mapping[str, str],
    *,
    expected_keys: Sequence[str],
    field: str,
) -> Mapping[str, str]:
    if not isinstance(value, Mapping) or set(value) != set(expected_keys):
        raise Stage13ProductionRuntimeError(f"{field} key set differs")
    result = {
        key: _require_hash(value[key], field=f"{field}.{key}") for key in expected_keys
    }
    return MappingProxyType(result)


def _stable_fixed_approval_receipt() -> Stage13ApprovalReceipt:
    """Read the one fixed approval path without following symlinks."""

    root = PROJECT_ROOT.absolute()
    path = STAGE13_APPROVAL_RECEIPT.absolute()
    try:
        path.relative_to(root)
    except ValueError as exc:  # pragma: no cover - constant construction guard
        raise Stage13ProductionRuntimeError(
            "fixed Stage 13 approval path escaped the project root"
        ) from exc
    current = root
    for index, component in enumerate(path.relative_to(root).parts):
        current = current / component
        try:
            metadata = current.stat(follow_symlinks=False)
        except OSError as exc:
            raise Stage13ProductionRuntimeError(
                "a separate fixed Stage 13 approval receipt is required"
            ) from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise Stage13ProductionRuntimeError(
                "fixed Stage 13 approval path contains a symlink"
            )
        final = index == len(path.relative_to(root).parts) - 1
        if final and not stat.S_ISREG(metadata.st_mode):
            raise Stage13ProductionRuntimeError(
                "fixed Stage 13 approval receipt is not a regular file"
            )
        if not final and not stat.S_ISDIR(metadata.st_mode):
            raise Stage13ProductionRuntimeError(
                "fixed Stage 13 approval parent is not a directory"
            )
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise Stage13ProductionRuntimeError(
            "fixed Stage 13 approval receipt could not be opened safely"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or before.st_size <= 0
            or before.st_size > 1024 * 1024
        ):
            raise Stage13ProductionRuntimeError(
                "fixed Stage 13 approval receipt has an unsafe identity"
            )
        chunks: list[bytes] = []
        while block := os.read(descriptor, 64 * 1024):
            chunks.append(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_nlink",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if any(getattr(before, name) != getattr(after, name) for name in identity):
        raise Stage13ProductionRuntimeError(
            "fixed Stage 13 approval receipt changed while read"
        )
    payload = b"".join(chunks)
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Stage13ProductionRuntimeError(
            "fixed Stage 13 approval receipt is invalid JSON"
        ) from exc
    if not isinstance(value, Mapping):
        raise Stage13ProductionRuntimeError(
            "fixed Stage 13 approval receipt must be an object"
        )
    try:
        receipt = Stage13ApprovalReceipt.from_mapping(value)
    except Stage13AuthorizationError as exc:
        raise Stage13ProductionRuntimeError(str(exc)) from exc
    if payload != _canonical_json_bytes(receipt.to_mapping()):
        raise Stage13ProductionRuntimeError(
            "fixed Stage 13 approval receipt is not canonical JSON"
        )
    return receipt


def _stable_bound_bytes(
    path: Path,
    *,
    trusted_root: Path,
    expected_sha256: str | None,
    expected_size_bytes: int | None = None,
    maximum_size_bytes: int | None = None,
    label: str,
) -> bytes:
    """Hash one fixed authority or payload without following symlinks."""

    if expected_sha256 is not None:
        _require_hash(expected_sha256, field=f"{label}_sha256")
    root = trusted_root.absolute()
    selected = path.absolute()
    try:
        relative = selected.relative_to(root)
    except ValueError as exc:
        raise Stage13ProductionRuntimeError(
            f"{label} escaped its trusted root"
        ) from exc
    current = root
    for index, component in enumerate(relative.parts):
        current = current / component
        try:
            metadata = current.stat(follow_symlinks=False)
        except OSError as exc:
            raise Stage13ProductionRuntimeError(f"{label} is absent") from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise Stage13ProductionRuntimeError(f"{label} contains a symlink")
        final = index == len(relative.parts) - 1
        if final and not stat.S_ISREG(metadata.st_mode):
            raise Stage13ProductionRuntimeError(f"{label} is not a regular file")
        if not final and not stat.S_ISDIR(metadata.st_mode):
            raise Stage13ProductionRuntimeError(f"{label} parent is not a directory")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(selected, flags)
    except OSError as exc:
        raise Stage13ProductionRuntimeError(
            f"{label} could not be opened safely"
        ) from exc
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_nlink != 1
            or (
                expected_size_bytes is not None
                and before.st_size != expected_size_bytes
            )
            or (maximum_size_bytes is not None and before.st_size > maximum_size_bytes)
        ):
            raise Stage13ProductionRuntimeError(f"{label} identity differs")
        digest = hashlib.sha256()
        chunks: list[bytes] = []
        while block := os.read(descriptor, 8 * 1024 * 1024):
            digest.update(block)
            chunks.append(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_nlink",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if any(getattr(before, name) != getattr(after, name) for name in identity):
        raise Stage13ProductionRuntimeError(f"{label} changed while read")
    if expected_sha256 is not None and digest.hexdigest() != expected_sha256:
        raise Stage13ProductionRuntimeError(f"{label} content digest differs")
    return b"".join(chunks)


def _json_mapping(payload: bytes, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Stage13ProductionRuntimeError(f"{label} is invalid JSON") from exc
    if not isinstance(value, Mapping):
        raise Stage13ProductionRuntimeError(f"{label} must be a JSON object")
    return value


@dataclass(frozen=True, slots=True)
class AuthenticatedPrefinalTrainingHistory:
    """Fixed Stage9+Stage11 feature/label history for Stage13 refits.

    This authority does not contain a Stage11 portfolio callback and cannot
    rerun or mutate Stage11.  It exposes only the historical predictor and
    completed-label projections needed by the final-specific refit adapter.
    """

    stage12_binding_sha256: str
    stage9_inputs: Stage9Inputs
    validation_predictor: AuthenticatedValidationPredictor
    development_oof: AuthenticatedDevelopmentOOF
    validation_outcome_path: Path
    validation_outcome_sha256: str
    validation_outcome_size_bytes: int
    validation_score_path: Path
    validation_score_sha256: str
    validation_score_size_bytes: int
    stage11_execution_input_manifest_sha256: str
    stage11_primary_manifest_sha256: str
    source_identity_sha256: str

    def __post_init__(self) -> None:
        for field in (
            "stage12_binding_sha256",
            "validation_outcome_sha256",
            "validation_score_sha256",
            "stage11_execution_input_manifest_sha256",
            "stage11_primary_manifest_sha256",
            "source_identity_sha256",
        ):
            _require_hash(getattr(self, field), field=field)
        if not isinstance(self.stage9_inputs, Stage9Inputs):
            raise Stage13ProductionRuntimeError(
                "Stage9 training authority type differs"
            )
        if not isinstance(self.validation_predictor, AuthenticatedValidationPredictor):
            raise Stage13ProductionRuntimeError(
                "Stage11 predictor authority type differs"
            )
        if not isinstance(self.development_oof, AuthenticatedDevelopmentOOF):
            raise Stage13ProductionRuntimeError("Stage9 OOF authority type differs")
        if (
            not self.validation_outcome_path.is_absolute()
            or self.validation_outcome_size_bytes <= 0
            or not self.validation_score_path.is_absolute()
            or self.validation_score_size_bytes <= 0
        ):
            raise Stage13ProductionRuntimeError(
                "Stage11 completed-label payload identity differs"
            )
        expected = _sha256_json(self.identity_payload())
        if self.source_identity_sha256 != expected:
            raise Stage13ProductionRuntimeError(
                "prefinal training source identity digest differs"
            )

    def identity_payload(self) -> dict[str, object]:
        return {
            "stage12_binding_sha256": self.stage12_binding_sha256,
            "stage9_input_receipts": [
                dict(value) for value in self.stage9_inputs.input_receipts
            ],
            "stage9_authority": dict(self.stage9_inputs.authority),
            "stage11_predictor_manifest_sha256": (
                self.validation_predictor.manifest_sha256
            ),
            "stage11_predictor_payload_sha256": (
                self.validation_predictor.registration.sha256
            ),
            "stage9_development_oof_sha256": self.development_oof.sha256,
            "stage9_development_oof_rows": self.development_oof.rows,
            "stage11_validation_outcome_sha256": self.validation_outcome_sha256,
            "stage11_validation_outcome_size_bytes": (
                self.validation_outcome_size_bytes
            ),
            "stage11_validation_score_sha256": self.validation_score_sha256,
            "stage11_validation_score_size_bytes": (self.validation_score_size_bytes),
            "stage11_execution_input_manifest_sha256": (
                self.stage11_execution_input_manifest_sha256
            ),
            "stage11_primary_build_id": STAGE11_PRIMARY_BUILD_ID,
            "stage11_primary_manifest_sha256": self.stage11_primary_manifest_sha256,
            "stage11_verification_id": STAGE11_VERIFICATION_ID,
            "maximum_prefinal_holding_month": "2019-11",
            "final_period_rows_permitted": 0,
        }

    def feature_dictionary(self, scope: FeatureScope) -> pa.Table:
        dictionary = self.stage9_inputs.read_feature_dictionary()
        if scope is FeatureScope.FULL_FEATURE_PRIMARY:
            return dictionary
        membership = self.stage9_inputs.read_feature_set_membership().to_pandas()
        included = frozenset(
            membership.loc[
                membership["feature_set"].eq("reduced_auxiliary_core")
                & membership["included"].astype(bool),
                "feature_name",
            ].astype(str)
        )
        frame = dictionary.to_pandas()
        selected = frame.loc[frame["feature_name"].astype(str).isin(included)].copy()
        if len(selected) != 111 or set(selected["feature_name"].astype(str)) != set(
            included
        ):
            raise Stage13ProductionRuntimeError(
                "authenticated reduced-core feature dictionary differs"
            )
        return pa.Table.from_pandas(selected, preserve_index=False)

    def feature_names(self, scope: FeatureScope) -> tuple[str, ...]:
        return tuple(
            str(value)
            for value in self.feature_dictionary(scope)["feature_name"].to_pylist()
        )

    def category_vocabulary(self, scope: FeatureScope) -> pa.Table:
        """Return the exact Stage 9 vocabulary for the requested frozen scope."""

        vocabulary = self.stage9_inputs.read_category_vocabulary()
        if scope is FeatureScope.FULL_FEATURE_PRIMARY:
            return vocabulary
        included = frozenset(self.feature_names(scope))
        frame = vocabulary.to_pandas()
        selected = frame.loc[frame["feature_name"].astype(str).isin(included)].copy()
        if selected.empty or not set(selected["feature_name"].astype(str)).issubset(
            included
        ):
            raise Stage13ProductionRuntimeError(
                "authenticated reduced-core category vocabulary differs"
            )
        return pa.Table.from_pandas(
            selected,
            schema=vocabulary.schema,
            preserve_index=False,
            safe=True,
        )

    def read_prefinal_features(self, scope: FeatureScope) -> pd.DataFrame:
        """Read only 2000--2019 predictor rows through fixed brokers."""

        feature_names = self.feature_names(scope)
        columns = tuple(dict.fromkeys((*FEATURE_SPINE_COLUMNS, *feature_names)))
        development = self.stage9_inputs.read_development_feature_panel(
            columns=columns
        ).to_pandas()
        record = self.validation_predictor.registration
        broker = self.validation_predictor.build_broker()
        request = broker.authorize_validation_table_request(
            self.validation_predictor.registration_name,
            role=PayloadRole.VALIDATION_PREDICTOR,
            columns=columns,
            minimum_signal_date=record.minimum_signal_date,
            maximum_signal_date=record.maximum_signal_date,
        )
        validation_table = broker.read_guarded_parquet(request)
        validation = validation_table.filter(
            pc.fill_null(validation_table["long_eligible"], False)
        ).to_pandas()
        combined = pd.concat((development, validation), ignore_index=True)
        combined["signal_date"] = pd.to_datetime(
            combined["signal_date"], errors="raise"
        ).dt.date
        combined["permno"] = pd.to_numeric(combined["permno"], errors="raise").astype(
            "int64"
        )
        if (
            tuple(combined.columns) != columns
            or combined.duplicated(["signal_date", "permno"]).any()
            or (combined["signal_date"] >= pd.Timestamp("2019-12-01").date()).any()
        ):
            raise Stage13ProductionRuntimeError(
                "prefinal predictor history escaped its frozen boundary"
            )
        return combined.sort_values(
            ["signal_date", "permno"], kind="mergesort"
        ).reset_index(drop=True)

    def aggregate_history(
        self,
        scope: FeatureScope,
        combined_features: pd.DataFrame,
    ) -> pd.DataFrame:
        """Build one exact aggregate-state row per authenticated signal date.

        Stage 9's dedicated aggregate reader retains the five development
        dates with no long-eligible security.  Later aggregate observations
        are collapsed only after proving that their replicated values are
        constant within a signal-date cross-section.
        """

        dictionary = self.feature_dictionary(scope).to_pandas()
        aggregate_names = tuple(
            dictionary.loc[
                dictionary["kind"].astype(str).eq("aggregate_state"),
                "feature_name",
            ].astype(str)
        )
        if not aggregate_names:
            raise Stage13ProductionRuntimeError(
                "authenticated feature scope omits aggregate-state predictors"
            )
        missing = sorted(
            {"signal_date", *aggregate_names} - set(combined_features.columns)
        )
        if missing:
            raise Stage13ProductionRuntimeError(
                "combined predictor history omits aggregate fields: "
                + ", ".join(missing)
            )
        later = combined_features.loc[:, ["signal_date", *aggregate_names]].copy()
        later["signal_date"] = pd.to_datetime(
            later["signal_date"], errors="raise"
        ).dt.date
        grouped = later.groupby("signal_date", sort=True, observed=True)
        if any(
            bool((grouped[name].nunique(dropna=False) > 1).any())
            for name in aggregate_names
        ):
            raise Stage13ProductionRuntimeError(
                "aggregate-state predictor varies within a signal date"
            )
        later = grouped[list(aggregate_names)].first().reset_index()
        development = self.stage9_inputs.read_aggregate_signal_history().to_pandas()
        development["signal_date"] = pd.to_datetime(
            development["signal_date"], errors="raise"
        ).dt.date
        development = development.loc[:, ["signal_date", *aggregate_names]]
        combined = pd.concat((development, later), ignore_index=True)
        for signal_date, rows in combined.groupby(
            "signal_date", sort=True, observed=True
        ):
            if any(rows[name].nunique(dropna=False) > 1 for name in aggregate_names):
                raise Stage13ProductionRuntimeError(
                    f"aggregate-state authority conflicts at {signal_date}"
                )
        result = (
            combined.groupby("signal_date", sort=True, observed=True)[
                list(aggregate_names)
            ]
            .first()
            .reset_index()
        )
        if result["signal_date"].duplicated().any():  # pragma: no cover - group guard
            raise Stage13ProductionRuntimeError(
                "aggregate-state history contains duplicate dates"
            )
        return result

    def read_prefinal_labels(self) -> pd.DataFrame:
        """Read development plus verified Stage11 completed-label evidence."""

        development = self.stage9_inputs.read_development_labels().to_pandas()
        development = development.rename(
            columns={"month_label_available_date": "outcome_availability_date"}
        )
        payload = _stable_bound_bytes(
            self.validation_outcome_path,
            trusted_root=STAGE11_DATABASE_ROOT,
            expected_sha256=self.validation_outcome_sha256,
            expected_size_bytes=self.validation_outcome_size_bytes,
            label="verified Stage11 completed-label payload",
        )
        try:
            import io

            validation = pq.read_table(
                io.BytesIO(payload),
                columns=[
                    "signal_date",
                    "holding_month",
                    "entry_date",
                    "exit_date",
                    "outcome_availability_date",
                    "permno",
                    "unranked_residual_return",
                    "residual_return_rank",
                    "primary_label_status",
                ],
                use_threads=False,
            ).to_pandas()
        except (pa.ArrowException, OSError) as exc:
            raise Stage13ProductionRuntimeError(
                "verified Stage11 completed-label payload could not be projected"
            ) from exc
        required = (
            "signal_date",
            "holding_month",
            "entry_date",
            "exit_date",
            "outcome_availability_date",
            "permno",
            "unranked_residual_return",
            "residual_return_rank",
            "primary_label_status",
        )
        development = development.loc[:, list(required)]
        combined = pd.concat((development, validation), ignore_index=True)
        for name in (
            "signal_date",
            "entry_date",
            "exit_date",
            "outcome_availability_date",
        ):
            combined[name] = pd.to_datetime(combined[name], errors="raise").dt.date
        combined["holding_month"] = combined["holding_month"].astype(str)
        combined["permno"] = pd.to_numeric(combined["permno"], errors="raise").astype(
            "int64"
        )
        if (
            combined.duplicated(["signal_date", "permno"]).any()
            or combined["holding_month"].max() > "2019-11"
            or (combined["signal_date"] >= pd.Timestamp("2019-12-01").date()).any()
        ):
            raise Stage13ProductionRuntimeError(
                "prefinal completed-label history escaped its frozen boundary"
            )
        return combined.sort_values(
            ["signal_date", "permno"], kind="mergesort"
        ).reset_index(drop=True)

    def read_prefinal_model_scores(self) -> pd.DataFrame:
        """Read the selected Stage9 OOF and Stage11 pseudo-live score history."""

        selected_columns = {
            family: self.development_oof.score_column_by_model[(family, config)]
            for family, config in (
                ("neural_network", "nn_001"),
                ("xgboost", "xgb_core_038_g10"),
                ("elastic_net", "enet_r004_l005"),
            )
        }
        oof_columns = (
            "signal_date",
            "holding_month",
            "permno",
            selected_columns["neural_network"],
            selected_columns["xgboost"],
            selected_columns["elastic_net"],
        )
        self.development_oof.gate()
        try:
            development = pq.read_table(
                self.development_oof.path,
                columns=list(oof_columns),
                use_threads=False,
            ).to_pandas()
        except (OSError, pa.ArrowException) as exc:
            raise Stage13ProductionRuntimeError(
                "authenticated Stage9 OOF scores could not be projected"
            ) from exc
        self.development_oof.gate()
        if len(development) != self.development_oof.rows:
            raise Stage13ProductionRuntimeError(
                "authenticated Stage9 OOF row count differs"
            )
        development["ensemble_score"] = 0.5 * pd.to_numeric(
            development[selected_columns["neural_network"]], errors="raise"
        ) + 0.5 * pd.to_numeric(
            development[selected_columns["xgboost"]], errors="raise"
        )
        development["elastic_net_score"] = pd.to_numeric(
            development[selected_columns["elastic_net"]], errors="raise"
        )
        development = development.loc[
            :,
            [
                "signal_date",
                "holding_month",
                "permno",
                "ensemble_score",
                "elastic_net_score",
            ],
        ]

        payload = _stable_bound_bytes(
            self.validation_score_path,
            trusted_root=STAGE11_DATABASE_ROOT,
            expected_sha256=self.validation_score_sha256,
            expected_size_bytes=self.validation_score_size_bytes,
            label="verified Stage11 selected-score payload",
        )
        try:
            import io

            validation = pq.read_table(
                io.BytesIO(payload),
                columns=[
                    "signal_date",
                    "holding_month",
                    "permno",
                    "ensemble_score",
                    "elastic_net_score",
                ],
                use_threads=False,
            ).to_pandas()
        except (OSError, pa.ArrowException) as exc:
            raise Stage13ProductionRuntimeError(
                "verified Stage11 selected scores could not be projected"
            ) from exc
        combined = pd.concat((development, validation), ignore_index=True)
        combined["signal_date"] = pd.to_datetime(
            combined["signal_date"], errors="raise"
        ).dt.date
        combined["holding_month"] = combined["holding_month"].astype(str)
        combined["permno"] = pd.to_numeric(combined["permno"], errors="raise").astype(
            "int64"
        )
        for name in ("ensemble_score", "elastic_net_score"):
            combined[name] = pd.to_numeric(combined[name], errors="raise")
        if (
            combined.empty
            or combined.duplicated(["holding_month", "permno"]).any()
            or combined["holding_month"].max() > "2019-11"
            or not np.isfinite(
                combined[["ensemble_score", "elastic_net_score"]].to_numpy(float)
            ).all()
            or (
                np.abs(
                    combined[["ensemble_score", "elastic_net_score"]].to_numpy(float)
                )
                > 0.5
            ).any()
        ):
            raise Stage13ProductionRuntimeError(
                "prefinal selected-score history differs"
            )
        return combined.sort_values(
            ["holding_month", "permno"], kind="mergesort"
        ).reset_index(drop=True)


def authenticate_prefinal_training_history(
    *,
    binding: Stage12VerifiedBinding,
    approval: Stage13ApprovalReceipt,
) -> AuthenticatedPrefinalTrainingHistory:
    """Authenticate fixed prefinal history after the separate Stage13 approval."""

    require_stage13_unblind_authorization(approval, expected_stage12_binding=binding)
    expected_input_sha = FROZEN_EXECUTION_INPUT_MANIFEST_SHA256
    if not isinstance(expected_input_sha, str):
        raise Stage13ProductionRuntimeError(
            "Stage11 execution-input manifest digest is not frozen"
        )
    input_payload = _stable_bound_bytes(
        FROZEN_EXECUTION_INPUT_MANIFEST_PATH,
        trusted_root=PROJECT_ROOT,
        expected_sha256=expected_input_sha,
        maximum_size_bytes=1024 * 1024,
        label="Stage11 execution-input manifest",
    )
    execution_inputs = _json_mapping(
        input_payload, label="Stage11 execution-input manifest"
    )
    predictor_binding = execution_inputs.get("predictor_authority")
    selector_binding = execution_inputs.get("development_selector")
    if (
        execution_inputs.get("manifest_id")
        != "phase05-stage11-empirical-execution-inputs-v1"
        or not isinstance(predictor_binding, Mapping)
        or not isinstance(selector_binding, Mapping)
        or selector_binding.get("build_id")
        != "phase04-selector-results-09a59d61ed58e77d80448a97"
    ):
        raise Stage13ProductionRuntimeError(
            "Stage11 prefinal execution-input authority differs"
        )
    predictor_path = Path(str(predictor_binding.get("manifest_path", "")))
    predictor = authenticate_validation_predictor_manifest(predictor_path)
    if predictor.manifest_sha256 != predictor_binding.get(
        "manifest_sha256"
    ) or predictor.registration.sha256 != predictor_binding.get("payload_sha256"):
        raise Stage13ProductionRuntimeError(
            "Stage11 predictor differs from its frozen execution-input binding"
        )
    stage9_inputs = resolve_stage9_inputs()
    stage9_receipts_sha256 = _sha256_json(
        [dict(value) for value in stage9_inputs.input_receipts]
    )
    if stage9_receipts_sha256 != selector_binding.get("input_receipts_sha256"):
        raise Stage13ProductionRuntimeError(
            "Stage9 authority differs from the Stage11 selector binding"
        )
    try:
        development_oof = _authenticate_development_oof(selector_binding)
    except Exception as exc:
        raise Stage13ProductionRuntimeError(
            "Stage9 OOF authority could not be authenticated"
        ) from exc
    if development_oof.stage9_inputs.input_receipts != stage9_inputs.input_receipts:
        raise Stage13ProductionRuntimeError("Stage9 OOF and feature authorities differ")

    pointer_payload = _stable_bound_bytes(
        STAGE11_EXECUTION_VERIFIED_POINTER,
        trusted_root=PROJECT_ROOT,
        expected_sha256=STAGE11_EXECUTION_VERIFIED_POINTER_SHA256,
        maximum_size_bytes=1024 * 1024,
        label="Stage11 verified execution pointer",
    )
    pointer = _json_mapping(pointer_payload, label="Stage11 verified execution pointer")
    authorizations = pointer.get("authorizations")
    if (
        pointer.get("status") != "STAGE11_VALIDATION_EXECUTION_VERIFIED"
        or pointer.get("difference_count") != 0
        or pointer.get("primary_build_id") != STAGE11_PRIMARY_BUILD_ID
        or pointer.get("primary_manifest_sha256") != STAGE11_PRIMARY_MANIFEST_SHA256
        or pointer.get("verification_id") != STAGE11_VERIFICATION_ID
        or not isinstance(authorizations, Mapping)
        or authorizations.get("final_2020_plus_access_authorized") is not False
    ):
        raise Stage13ProductionRuntimeError(
            "Stage11 verified execution authority differs"
        )
    primary_root = Path(str(pointer.get("primary_bundle_path", ""))).absolute()
    try:
        primary_root.relative_to(STAGE11_DATABASE_ROOT.absolute())
    except ValueError as exc:
        raise Stage13ProductionRuntimeError(
            "Stage11 primary bundle escaped its fixed database root"
        ) from exc
    if primary_root.name != STAGE11_PRIMARY_BUILD_ID:
        raise Stage13ProductionRuntimeError("Stage11 primary bundle ID differs")
    try:
        names = frozenset(item.name for item in primary_root.iterdir())
    except OSError as exc:
        raise Stage13ProductionRuntimeError("Stage11 primary bundle is absent") from exc
    if names != STAGE11_PRIMARY_MEMBERS:
        raise Stage13ProductionRuntimeError("Stage11 primary member set differs")
    manifest_payload = _stable_bound_bytes(
        primary_root / "manifest.json",
        trusted_root=STAGE11_DATABASE_ROOT,
        expected_sha256=STAGE11_PRIMARY_MANIFEST_SHA256,
        maximum_size_bytes=16 * 1024 * 1024,
        label="Stage11 primary manifest",
    )
    manifest = _json_mapping(manifest_payload, label="Stage11 primary manifest")
    members = manifest.get("members")
    outcome_receipt = (
        None
        if not isinstance(members, Mapping)
        else members.get(STAGE11_OUTCOME_MEMBER)
    )
    score_receipt = (
        None
        if not isinstance(members, Mapping)
        else members.get(STAGE11_SELECTED_SCORE_MEMBER)
    )
    if (
        manifest.get("build_id") != STAGE11_PRIMARY_BUILD_ID
        or manifest.get("status") != "STAGE11_VALIDATION_EXECUTION_COMPLETE"
        or not isinstance(outcome_receipt, Mapping)
        or set(outcome_receipt) != {"sha256", "size_bytes"}
        or not isinstance(outcome_receipt.get("size_bytes"), int)
        or not isinstance(score_receipt, Mapping)
        or set(score_receipt) != {"sha256", "size_bytes"}
        or not isinstance(score_receipt.get("size_bytes"), int)
    ):
        raise Stage13ProductionRuntimeError(
            "Stage11 completed-label member receipt differs"
        )
    values = {
        "stage12_binding_sha256": binding.binding_sha256,
        "stage9_inputs": stage9_inputs,
        "validation_predictor": predictor,
        "development_oof": development_oof,
        "validation_outcome_path": primary_root / STAGE11_OUTCOME_MEMBER,
        "validation_outcome_sha256": _require_hash(
            outcome_receipt.get("sha256"), field="validation_outcome_sha256"
        ),
        "validation_outcome_size_bytes": int(outcome_receipt["size_bytes"]),
        "validation_score_path": primary_root / STAGE11_SELECTED_SCORE_MEMBER,
        "validation_score_sha256": _require_hash(
            score_receipt.get("sha256"), field="validation_score_sha256"
        ),
        "validation_score_size_bytes": int(score_receipt["size_bytes"]),
        "stage11_execution_input_manifest_sha256": expected_input_sha,
        "stage11_primary_manifest_sha256": STAGE11_PRIMARY_MANIFEST_SHA256,
    }
    provisional = AuthenticatedPrefinalTrainingHistory.__new__(
        AuthenticatedPrefinalTrainingHistory
    )
    for name, value in (*values.items(), ("source_identity_sha256", "")):
        object.__setattr__(provisional, name, value)
    digest = _sha256_json(provisional.identity_payload())
    return AuthenticatedPrefinalTrainingHistory(**values, source_identity_sha256=digest)


@dataclass(frozen=True, slots=True)
class Stage13RefitIdentity:
    """Honest final-specific identity for one completed-label-only refit."""

    feature_scope: FeatureScope
    holding_month: str
    fit_signal_date: object
    prefinal_training_source_sha256: str
    final_prior_predictor_set_sha256: str
    completed_label_set_sha256: str
    completed_label_months: tuple[str, ...]
    maximum_outcome_available_date: object
    final_access_ledger_sha256: str
    training_data_sha256: str
    full_base_preprocessor_sha256: str
    full_elastic_preprocessor_sha256: str
    identity_sha256: str

    def __post_init__(self) -> None:
        import datetime as dt

        if not isinstance(self.feature_scope, FeatureScope):
            raise Stage13ProductionRuntimeError("refit feature scope differs")
        if type(self.fit_signal_date) is not dt.date or not is_quarterly_refit(
            holding_month=self.holding_month,
            signal_date=self.fit_signal_date,
        ):
            raise Stage13ProductionRuntimeError("refit chronology differs")
        if (
            type(self.maximum_outcome_available_date) is not dt.date
            or self.maximum_outcome_available_date > self.fit_signal_date
        ):
            raise Stage13ProductionRuntimeError(
                "refit used an unavailable completed label"
            )
        months = tuple(self.completed_label_months)
        if (
            not months
            or months != tuple(sorted(set(months)))
            or any(month >= self.holding_month for month in months)
        ):
            raise Stage13ProductionRuntimeError(
                "refit completed-label month set differs"
            )
        for field in (
            "prefinal_training_source_sha256",
            "final_prior_predictor_set_sha256",
            "completed_label_set_sha256",
            "final_access_ledger_sha256",
            "training_data_sha256",
            "full_base_preprocessor_sha256",
            "full_elastic_preprocessor_sha256",
        ):
            _require_hash(getattr(self, field), field=field)
        object.__setattr__(self, "completed_label_months", months)
        _require_hash(self.identity_sha256, field="identity_sha256")
        if self.identity_sha256 != _sha256_json(self.payload(False)):
            raise Stage13ProductionRuntimeError("final refit identity digest differs")

    def payload(self, include_digest: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "schema_version": 1,
            "stage": 13,
            "identity_kind": "final_specific_completed_label_quarterly_refit",
            "feature_scope": self.feature_scope.value,
            "holding_month": self.holding_month,
            "fit_signal_date": self.fit_signal_date.isoformat(),
            "prefinal_training_source_sha256": (self.prefinal_training_source_sha256),
            "final_prior_predictor_set_sha256": (self.final_prior_predictor_set_sha256),
            "completed_label_set_sha256": self.completed_label_set_sha256,
            "completed_label_months": list(self.completed_label_months),
            "maximum_outcome_available_date": (
                self.maximum_outcome_available_date.isoformat()
            ),
            "final_access_ledger_sha256": self.final_access_ledger_sha256,
            "training_data_sha256": self.training_data_sha256,
            "full_base_preprocessor_sha256": (self.full_base_preprocessor_sha256),
            "full_elastic_preprocessor_sha256": (self.full_elastic_preprocessor_sha256),
            "economic_kernel_contract_sha256": (
                FROZEN_FINAL_ECONOMIC_KERNEL_CONTRACT.contract_sha256
            ),
            "selected_model_grid": [
                definition.to_mapping()
                for definition in FROZEN_FINAL_ECONOMIC_KERNEL_CONTRACT.models
            ],
            "completed_labels_only": True,
            "stage11_receipt_republished_as_final": False,
        }
        if include_digest:
            value["identity_sha256"] = self.identity_sha256
        return value

    @classmethod
    def create(cls, **values: Any) -> "Stage13RefitIdentity":
        provisional = cls.__new__(cls)
        for name, value in (*values.items(), ("identity_sha256", "")):
            object.__setattr__(provisional, name, value)
        digest = _sha256_json(provisional.payload(False))
        return cls(**values, identity_sha256=digest)


def _frame_sha256(frame: pd.DataFrame) -> str:
    """Hash an ordered frame through metadata-free Arrow record batches."""

    table = pa.Table.from_pandas(frame, preserve_index=False).replace_schema_metadata()
    digest = hashlib.sha256()
    digest.update(table.schema.serialize().to_pybytes())
    for batch in table.combine_chunks().to_batches(max_chunksize=65_536):
        digest.update(batch.serialize().to_pybytes())
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class PreparedStage13NativeRefit:
    """Final-specific evidence plus the arrays accepted by the native fitters."""

    identity: Stage13RefitIdentity
    prepared: PreparedRefit

    def __post_init__(self) -> None:
        if not isinstance(self.identity, Stage13RefitIdentity) or not isinstance(
            self.prepared, PreparedRefit
        ):
            raise Stage13ProductionRuntimeError("prepared final refit type differs")
        if self.prepared.refit_signal_date != self.identity.fit_signal_date:
            raise Stage13ProductionRuntimeError(
                "prepared final refit chronology differs"
            )
        training_sha = _sha256_json(self.prepared.base_arrays.identity_payload())
        if (
            training_sha != self.identity.training_data_sha256
            or self.prepared.full_base_preprocessor.sha256
            != self.identity.full_base_preprocessor_sha256
            or self.prepared.full_elastic_preprocessor.sha256
            != self.identity.full_elastic_preprocessor_sha256
        ):
            raise Stage13ProductionRuntimeError(
                "prepared final refit evidence differs from its identity"
            )


def _normalized_final_feature_history(
    history: AuthenticatedPrefinalTrainingHistory,
    *,
    scope: FeatureScope,
    final_features: pd.DataFrame,
) -> pd.DataFrame:
    names = history.feature_names(scope)
    columns = tuple(dict.fromkeys((*FEATURE_SPINE_COLUMNS, *names)))
    missing = sorted(set(columns) - set(final_features.columns))
    if missing:
        raise Stage13ProductionRuntimeError(
            "final predictor history omits frozen fields: " + ", ".join(missing)
        )
    result = final_features.loc[:, list(columns)].copy()
    result["signal_date"] = pd.to_datetime(
        result["signal_date"], errors="raise"
    ).dt.date
    result["permno"] = pd.to_numeric(result["permno"], errors="raise").astype("int64")
    if (
        result.empty
        or result.duplicated(["signal_date", "permno"]).any()
        or (result["signal_date"] < dt.date(2019, 12, 1)).any()
    ):
        raise Stage13ProductionRuntimeError("final predictor history boundary differs")
    return result.sort_values(["signal_date", "permno"], kind="mergesort").reset_index(
        drop=True
    )


def _normalized_final_label_history(final_labels: pd.DataFrame) -> pd.DataFrame:
    columns = (
        "signal_date",
        "holding_month",
        "entry_date",
        "exit_date",
        "outcome_availability_date",
        "permno",
        "unranked_residual_return",
        "residual_return_rank",
        "primary_label_status",
    )
    missing = sorted(set(columns) - set(final_labels.columns))
    if missing:
        raise Stage13ProductionRuntimeError(
            "final completed-label history omits frozen fields: " + ", ".join(missing)
        )
    result = final_labels.loc[:, list(columns)].copy()
    for name in (
        "signal_date",
        "entry_date",
        "exit_date",
        "outcome_availability_date",
    ):
        result[name] = pd.to_datetime(result[name], errors="raise").dt.date
    result["holding_month"] = result["holding_month"].astype(str)
    result["permno"] = pd.to_numeric(result["permno"], errors="raise").astype("int64")
    if (
        result.duplicated(["signal_date", "permno"]).any()
        or (result["signal_date"] < dt.date(2019, 12, 1)).any()
    ):
        raise Stage13ProductionRuntimeError(
            "final completed-label history boundary differs"
        )
    return result.sort_values(["signal_date", "permno"], kind="mergesort").reset_index(
        drop=True
    )


def prepare_stage13_native_refit(
    *,
    history: AuthenticatedPrefinalTrainingHistory,
    scope: FeatureScope,
    holding_month: str,
    fit_signal_date: dt.date,
    final_features: pd.DataFrame,
    final_completed_labels: pd.DataFrame,
    final_access_ledger_sha256: str,
    scratch_root: Path,
    authenticated_prefinal_features: pd.DataFrame | None = None,
    authenticated_prefinal_labels: pd.DataFrame | None = None,
) -> PreparedStage13NativeRefit:
    """Prepare one honest completed-label-only Stage 13 quarterly refit."""

    if not isinstance(history, AuthenticatedPrefinalTrainingHistory):
        raise Stage13ProductionRuntimeError("prefinal training authority type differs")
    if not is_quarterly_refit(
        holding_month=holding_month,
        signal_date=fit_signal_date,
    ):
        raise Stage13ProductionRuntimeError("final native refit is not quarterly")
    _require_hash(
        final_access_ledger_sha256,
        field="final_access_ledger_sha256",
    )
    if (
        not isinstance(scratch_root, Path)
        or not scratch_root.is_absolute()
        or scratch_root.exists()
    ):
        raise Stage13ProductionRuntimeError(
            "final refit scratch root must be a fresh absolute path"
        )

    prefinal_features = (
        history.read_prefinal_features(scope)
        if authenticated_prefinal_features is None
        else authenticated_prefinal_features.copy()
    )
    expected_prefinal_columns = tuple(
        dict.fromkeys((*FEATURE_SPINE_COLUMNS, *history.feature_names(scope)))
    )
    if (
        tuple(prefinal_features.columns) != expected_prefinal_columns
        or prefinal_features.empty
        or prefinal_features.duplicated(["signal_date", "permno"]).any()
    ):
        raise Stage13ProductionRuntimeError(
            "cached authenticated prefinal predictor history differs"
        )
    observed_final_features = _normalized_final_feature_history(
        history,
        scope=scope,
        final_features=final_features,
    )
    scoring = observed_final_features.loc[
        observed_final_features["signal_date"].eq(fit_signal_date)
    ]
    if scoring.empty:
        raise Stage13ProductionRuntimeError(
            "current final scoring predictor cross-section is absent"
        )
    combined_features = pd.concat(
        (prefinal_features, observed_final_features), ignore_index=True
    ).sort_values(["signal_date", "permno"], kind="mergesort")
    if combined_features.duplicated(["signal_date", "permno"]).any():
        raise Stage13ProductionRuntimeError(
            "prefinal and final predictor histories overlap"
        )

    prefinal_labels = (
        history.read_prefinal_labels()
        if authenticated_prefinal_labels is None
        else authenticated_prefinal_labels.copy()
    )
    if (
        prefinal_labels.empty
        or prefinal_labels.duplicated(["signal_date", "permno"]).any()
        or prefinal_labels["holding_month"].astype(str).max() > "2019-11"
    ):
        raise Stage13ProductionRuntimeError(
            "cached authenticated prefinal completed-label history differs"
        )
    normalized_final_labels = _normalized_final_label_history(final_completed_labels)
    labels = pd.concat(
        (prefinal_labels, normalized_final_labels), ignore_index=True
    ).sort_values(["holding_month", "permno"], kind="mergesort")
    if labels.duplicated(["signal_date", "permno"]).any():
        raise Stage13ProductionRuntimeError(
            "prefinal and final completed-label histories overlap"
        )
    rank = pd.to_numeric(labels["residual_return_rank"], errors="coerce").to_numpy(
        float
    )
    completed = (
        labels["primary_label_status"].astype(str).eq("PRESENT").to_numpy(bool)
        & np.isfinite(rank)
        & labels["exit_date"].le(fit_signal_date).to_numpy(bool)
        & labels["outcome_availability_date"].le(fit_signal_date).to_numpy(bool)
    )
    admitted = labels.loc[completed].copy()
    admitted["residual_return_rank"] = rank[completed]
    if admitted.empty:
        raise Stage13ProductionRuntimeError("final refit completed-label set is empty")
    final_prior = observed_final_features.loc[
        observed_final_features["signal_date"] < fit_signal_date
    ].reset_index(drop=True)
    prepared = prepare_authenticated_refit(
        refit_signal_date=fit_signal_date,
        scoring_signal_date=fit_signal_date,
        combined_features=combined_features,
        aggregate_history=history.aggregate_history(scope, combined_features),
        training_labels=labels,
        feature_dictionary=history.feature_dictionary(scope),
        category_vocabulary=history.category_vocabulary(scope),
        stage9_inputs=history.stage9_inputs,
        scratch_root=scratch_root,
        permit_projected_global_ordinals=(scope is FeatureScope.REDUCED_AUXILIARY_CORE),
    )
    identity = Stage13RefitIdentity.create(
        feature_scope=scope,
        holding_month=holding_month,
        fit_signal_date=fit_signal_date,
        prefinal_training_source_sha256=history.source_identity_sha256,
        final_prior_predictor_set_sha256=_frame_sha256(final_prior),
        completed_label_set_sha256=_frame_sha256(admitted),
        completed_label_months=tuple(
            sorted(set(admitted["holding_month"].astype(str)))
        ),
        maximum_outcome_available_date=max(admitted["outcome_availability_date"]),
        final_access_ledger_sha256=final_access_ledger_sha256,
        training_data_sha256=_sha256_json(prepared.base_arrays.identity_payload()),
        full_base_preprocessor_sha256=prepared.full_base_preprocessor.sha256,
        full_elastic_preprocessor_sha256=(prepared.full_elastic_preprocessor.sha256),
    )
    return PreparedStage13NativeRefit(identity=identity, prepared=prepared)


@dataclass(frozen=True, slots=True)
class Stage13NativeFamilyState:
    """Restorable final-specific native state for one scope/model family."""

    feature_scope: FeatureScope
    family: str
    config_id: str
    seeds: tuple[int | None, ...]
    refit_identity_sha256: str
    technical_identity_sha256_by_seed: tuple[str, ...]
    model_format: str
    model_payload_by_seed: tuple[bytes, ...]
    full_preprocessor: PersistedPreprocessorState
    inner_preprocessor: PersistedPreprocessorState | None
    state_sha256: str

    def __post_init__(self) -> None:
        definition = next(
            (
                value
                for value in FROZEN_FINAL_ECONOMIC_KERNEL_CONTRACT.models
                if value.family == self.family
            ),
            None,
        )
        if (
            definition is None
            or self.config_id != definition.config_id
            or self.seeds != definition.seeds
            or len(self.technical_identity_sha256_by_seed) != len(self.seeds)
            or len(self.model_payload_by_seed) != len(self.seeds)
            or any(
                not isinstance(payload, bytes) or not payload
                for payload in self.model_payload_by_seed
            )
            or not isinstance(self.full_preprocessor, PersistedPreprocessorState)
        ):
            raise Stage13ProductionRuntimeError("final native family state differs")
        if self.family == "elastic_net":
            if (
                self.inner_preprocessor is not None
                or self.full_preprocessor.matrix_kind != "elastic"
            ):
                raise Stage13ProductionRuntimeError(
                    "final Elastic Net preprocessing state differs"
                )
        elif (
            not isinstance(self.inner_preprocessor, PersistedPreprocessorState)
            or self.full_preprocessor.matrix_kind != "base"
            or self.inner_preprocessor.matrix_kind != "base"
        ):
            raise Stage13ProductionRuntimeError(
                "final stochastic preprocessing state differs"
            )
        _require_hash(self.refit_identity_sha256, field="refit_identity_sha256")
        for value in self.technical_identity_sha256_by_seed:
            _require_hash(value, field="technical_identity_sha256")
        if not isinstance(self.model_format, str) or not self.model_format:
            raise Stage13ProductionRuntimeError("final native model format is absent")
        _require_hash(self.state_sha256, field="state_sha256")
        if self.state_sha256 != _sha256_json(self.payload(False, False)):
            raise Stage13ProductionRuntimeError("final native family digest differs")

    @property
    def model_payload_sha256_by_seed(self) -> tuple[str, ...]:
        return tuple(_sha256_bytes(value) for value in self.model_payload_by_seed)

    def payload(
        self,
        include_digest: bool = True,
        include_model_payload: bool = False,
    ) -> dict[str, object]:
        value: dict[str, object] = {
            "schema_version": 1,
            "stage": 13,
            "feature_scope": self.feature_scope.value,
            "family": self.family,
            "config_id": self.config_id,
            "seeds": list(self.seeds),
            "refit_identity_sha256": self.refit_identity_sha256,
            "technical_identity_sha256_by_seed": list(
                self.technical_identity_sha256_by_seed
            ),
            "model_format": self.model_format,
            "model_payload_sha256_by_seed": list(self.model_payload_sha256_by_seed),
            "full_preprocessor": self.full_preprocessor.payload(),
            "inner_preprocessor": (
                None
                if self.inner_preprocessor is None
                else self.inner_preprocessor.payload()
            ),
            "stage11_fit_receipt_republished_as_final": False,
        }
        if include_model_payload:
            value["model_payload_base64_by_seed"] = [
                base64.b64encode(payload).decode("ascii")
                for payload in self.model_payload_by_seed
            ]
        if include_digest:
            value["state_sha256"] = self.state_sha256
        return value

    @classmethod
    def create(cls, **values: Any) -> "Stage13NativeFamilyState":
        provisional = cls.__new__(cls)
        for name, value in (*values.items(), ("state_sha256", "")):
            object.__setattr__(provisional, name, value)
        digest = _sha256_json(provisional.payload(False, False))
        return cls(**values, state_sha256=digest)

    @classmethod
    def from_payload(cls, value: Mapping[str, object]) -> "Stage13NativeFamilyState":
        expected = {
            "schema_version",
            "stage",
            "feature_scope",
            "family",
            "config_id",
            "seeds",
            "refit_identity_sha256",
            "technical_identity_sha256_by_seed",
            "model_format",
            "model_payload_sha256_by_seed",
            "model_payload_base64_by_seed",
            "full_preprocessor",
            "inner_preprocessor",
            "stage11_fit_receipt_republished_as_final",
            "state_sha256",
        }
        if (
            set(value) != expected
            or value.get("schema_version") != 1
            or value.get("stage") != 13
            or value.get("stage11_fit_receipt_republished_as_final") is not False
            or not isinstance(value.get("full_preprocessor"), Mapping)
            or not isinstance(value.get("model_payload_base64_by_seed"), list)
            or not isinstance(value.get("model_payload_sha256_by_seed"), list)
        ):
            raise Stage13ProductionRuntimeError(
                "persisted final native state fields differ"
            )
        try:
            scope = FeatureScope(str(value["feature_scope"]))
            payloads = tuple(
                base64.b64decode(str(item), validate=True)
                for item in value["model_payload_base64_by_seed"]
            )
        except (ValueError, TypeError) as exc:
            raise Stage13ProductionRuntimeError(
                "persisted final native state payload differs"
            ) from exc
        observed_hashes = tuple(_sha256_bytes(item) for item in payloads)
        if observed_hashes != tuple(value["model_payload_sha256_by_seed"]):
            raise Stage13ProductionRuntimeError(
                "persisted final native model bytes differ"
            )
        inner_payload = value["inner_preprocessor"]
        if inner_payload is not None and not isinstance(inner_payload, Mapping):
            raise Stage13ProductionRuntimeError(
                "persisted final inner preprocessor differs"
            )
        return cls(
            feature_scope=scope,
            family=str(value["family"]),
            config_id=str(value["config_id"]),
            seeds=tuple(value["seeds"]),  # type: ignore[arg-type]
            refit_identity_sha256=str(value["refit_identity_sha256"]),
            technical_identity_sha256_by_seed=tuple(
                str(item) for item in value["technical_identity_sha256_by_seed"]  # type: ignore[union-attr]
            ),
            model_format=str(value["model_format"]),
            model_payload_by_seed=payloads,
            full_preprocessor=PersistedPreprocessorState.from_payload(
                value["full_preprocessor"]  # type: ignore[arg-type]
            ),
            inner_preprocessor=(
                None
                if inner_payload is None
                else PersistedPreprocessorState.from_payload(inner_payload)
            ),
            state_sha256=str(value["state_sha256"]),
        )


def _native_fit_family_worker(
    identities: tuple[ValidationRefitIdentity, ...],
    request: object,
    elastic_scratch_path: str | None,
    result_queue: object,
) -> None:
    """Fit exactly one family in a spawned native worker."""

    try:
        from portfolio_bot.validation.selector_native_runtime import (
            fit_locked_selector_model_from_paths,
        )

        if not identities:
            raise Stage13ProductionRuntimeError("native fit worker family is absent")
        family = identities[0].family
        if any(value.family != family for value in identities):
            raise Stage13ProductionRuntimeError("native fit worker family differs")
        artifacts = tuple(
            fit_locked_selector_model_from_paths(
                identity=identity,
                request=request,  # type: ignore[arg-type]
                elastic_scratch_path=(
                    None if elastic_scratch_path is None else Path(elastic_scratch_path)
                ),
            )
            for identity in identities
        )
        # Stage 11 fit receipts intentionally are not returned.  The final
        # caller publishes only Stage13RefitIdentity and payload hashes.
        result_queue.put(  # type: ignore[attr-defined]
            {
                "ok": True,
                "model_format": artifacts[0].model_format,
                "model_payloads": tuple(value.model_payload for value in artifacts),
            }
        )
    except BaseException:
        result_queue.put(  # type: ignore[attr-defined]
            {"ok": False, "traceback": traceback.format_exc()}
        )


def _run_spawned_worker(
    *,
    target: Callable[..., None],
    args: tuple[object, ...],
    name: str,
) -> Mapping[str, Any]:
    context = mp.get_context("spawn")
    queue = context.Queue()
    process = context.Process(target=target, args=(*args, queue), name=name)
    process.start()
    try:
        # Drain the result before joining.  ``multiprocessing.Queue`` uses a
        # feeder pipe; a child returning a large model payload can otherwise
        # block during process shutdown while the parent waits in ``join``.
        while True:
            try:
                message = queue.get(timeout=1.0)
                break
            except Empty:
                if process.is_alive():
                    continue
                process.join()
                try:
                    message = queue.get(timeout=10.0)
                    break
                except Empty as exc:
                    raise Stage13ProductionRuntimeError(
                        f"native worker {name} exited without a receipt"
                    ) from exc
        process.join(timeout=10.0)
        if process.is_alive():
            raise Stage13ProductionRuntimeError(
                f"native worker {name} did not exit after its receipt was drained"
            )
        if (
            process.exitcode != 0
            or not isinstance(message, Mapping)
            or message.get("ok") is not True
        ):
            details = (
                message.get("traceback", "no traceback")
                if isinstance(message, Mapping)
                else "invalid receipt"
            )
            raise Stage13ProductionRuntimeError(
                f"native worker {name} failed: {details}"
            )
        return MappingProxyType(dict(message))
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=10.0)
        if process.is_alive():
            process.kill()
            process.join()
        queue.close()


def fit_stage13_selected_native_models(
    refit: PreparedStage13NativeRefit,
    *,
    work_root: Path,
) -> Mapping[str, Stage13NativeFamilyState]:
    """Run the selected NN/XGB/ENet fit grid with final-specific receipts."""

    if not isinstance(refit, PreparedStage13NativeRefit):
        raise Stage13ProductionRuntimeError("final native refit type differs")
    if (
        not isinstance(work_root, Path)
        or not work_root.is_absolute()
        or work_root.exists()
    ):
        raise Stage13ProductionRuntimeError(
            "native fit work root must be a fresh absolute path"
        )
    work_root.mkdir(mode=0o700, parents=True, exist_ok=False)
    results: dict[str, Stage13NativeFamilyState] = {}
    try:
        for definition in FROZEN_FINAL_ECONOMIC_KERNEL_CONTRACT.models:
            family = definition.family
            arrays = (
                refit.prepared.elastic_arrays
                if family == "elastic_net"
                else refit.prepared.base_arrays
            )
            full_preprocessor = (
                refit.prepared.full_elastic_preprocessor
                if family == "elastic_net"
                else refit.prepared.full_base_preprocessor
            )
            inner_preprocessor = (
                None if family == "elastic_net" else refit.prepared.inner_preprocessor
            )
            identities = tuple(
                build_refit_identity(
                    family=family,
                    config_id=definition.config_id,
                    seed=seed,
                    refit_signal_date=refit.identity.fit_signal_date,
                    arrays=arrays,
                    full_preprocessor=full_preprocessor,
                    inner_preprocessor=inner_preprocessor,
                    validation_calendar_sha256=_sha256_json(
                        {
                            "stage": 13,
                            "holding_month": refit.identity.holding_month,
                            "fit_signal_date": (
                                refit.identity.fit_signal_date.isoformat()
                            ),
                            "stage12_binding": (
                                refit.identity.prefinal_training_source_sha256
                            ),
                        }
                    ),
                    model_code_sha256=_model_code_sha256(family),
                    runtime_versions_sha256=RUNTIME_VERSIONS_SHA256,
                    stage9_shortlist_sha256=LOCKED_STAGE9_SHORTLIST_SHA256,
                )
                for seed in definition.seeds
            )
            request = persist_validation_refit_arrays_for_spawn(
                arrays,
                work_root / f"{family}-arrays",
            )
            message = _run_spawned_worker(
                target=_native_fit_family_worker,
                args=(
                    identities,
                    request,
                    (
                        str(work_root / "elastic-net-scratch")
                        if family == "elastic_net"
                        else None
                    ),
                ),
                name=(
                    f"stage13-{refit.identity.feature_scope.value}-{family}-"
                    f"{refit.identity.holding_month}"
                ),
            )
            payloads = message.get("model_payloads")
            if (
                not isinstance(payloads, tuple)
                or len(payloads) != len(definition.seeds)
                or not all(isinstance(value, bytes) for value in payloads)
            ):
                raise Stage13ProductionRuntimeError(
                    "native final fit returned an invalid model set"
                )
            results[family] = Stage13NativeFamilyState.create(
                feature_scope=refit.identity.feature_scope,
                family=family,
                config_id=definition.config_id,
                seeds=definition.seeds,
                refit_identity_sha256=refit.identity.identity_sha256,
                technical_identity_sha256_by_seed=tuple(
                    value.sha256 for value in identities
                ),
                model_format=str(message["model_format"]),
                model_payload_by_seed=payloads,
                full_preprocessor=full_preprocessor,
                inner_preprocessor=inner_preprocessor,
            )
    finally:
        # The authenticated NPY requests are scratch, never authority.  Model
        # bytes are returned in the content-addressed family states.
        import shutil

        shutil.rmtree(work_root, ignore_errors=True)
    if set(results) != {
        value.family for value in FROZEN_FINAL_ECONOMIC_KERNEL_CONTRACT.models
    }:
        raise Stage13ProductionRuntimeError("native final model family set differs")
    return MappingProxyType(results)


def _native_predict_family_worker(
    state_payload: Mapping[str, object],
    score_request: object,
    result_queue: object,
) -> None:
    try:
        from portfolio_bot.validation.selector_native_runtime import (
            predict_locked_selector_model,
        )

        state = Stage13NativeFamilyState.from_payload(state_payload)
        matrix = load_array_for_spawn(score_request)  # type: ignore[arg-type]
        forecasts: list[np.ndarray] = []
        for seed, payload, technical_sha in zip(
            state.seeds,
            state.model_payload_by_seed,
            state.technical_identity_sha256_by_seed,
            strict=True,
        ):
            checkpoint = LoadedSelectorCheckpoint(
                root=Path("."),
                manifest={"checkpoint_id": f"stage13-native-{technical_sha[:24]}"},
                identity={
                    "family": state.family,
                    "config_id": state.config_id,
                    "seed": seed,
                },
                model_payload=payload,
                inner_preprocessor=state.inner_preprocessor,
                full_preprocessor=state.full_preprocessor,
            )
            forecasts.append(
                predict_locked_selector_model(
                    checkpoint=checkpoint,
                    full_frozen_matrix=matrix,
                )
            )
        result_queue.put(  # type: ignore[attr-defined]
            {"ok": True, "forecasts": tuple(forecasts)}
        )
    except BaseException:
        result_queue.put(  # type: ignore[attr-defined]
            {"ok": False, "traceback": traceback.format_exc()}
        )


def predict_stage13_selected_native_models(
    *,
    states: Mapping[str, Stage13NativeFamilyState],
    holding_month: str,
    feature_scope: FeatureScope,
    permno: np.ndarray,
    score_base_frozen: np.ndarray,
    score_elastic_frozen: np.ndarray,
    work_root: Path,
) -> FinalForecastReceipt:
    """Score all selected seeds and apply the frozen final aggregation."""

    definitions = {
        value.family: value for value in FROZEN_FINAL_ECONOMIC_KERNEL_CONTRACT.models
    }
    if set(states) != set(definitions):
        raise Stage13ProductionRuntimeError("active native model family set differs")
    if work_root.exists() or not work_root.is_absolute():
        raise Stage13ProductionRuntimeError(
            "native score work root must be a fresh absolute path"
        )
    work_root.mkdir(mode=0o700, parents=True, exist_ok=False)
    blocks: dict[str, SeedForecastBlock] = {}
    try:
        for family, definition in definitions.items():
            state = states[family]
            if state.feature_scope is not feature_scope:
                raise Stage13ProductionRuntimeError(
                    "active native model feature scope differs"
                )
            matrix = (
                score_elastic_frozen if family == "elastic_net" else score_base_frozen
            )
            request = persist_array_for_spawn(
                matrix,
                work_root / f"{family}-score",
                name="score_matrix",
            )
            message = _run_spawned_worker(
                target=_native_predict_family_worker,
                args=(state.payload(include_model_payload=True), request),
                name=f"stage13-score-{feature_scope.value}-{family}-{holding_month}",
            )
            forecasts = message.get("forecasts")
            if not isinstance(forecasts, tuple) or len(forecasts) != len(
                definition.seeds
            ):
                raise Stage13ProductionRuntimeError(
                    "native final scoring result set differs"
                )
            blocks[family] = SeedForecastBlock.create(
                model_family=family,
                config_id=definition.config_id,
                seeds=definition.seeds,
                raw_forecasts=np.vstack(forecasts),
                fitted_state_sha256_by_seed=(state.model_payload_sha256_by_seed),
            )
    finally:
        import shutil

        shutil.rmtree(work_root, ignore_errors=True)
    return aggregate_selected_forecasts(
        feature_scope=feature_scope,
        holding_month=holding_month,
        permno=permno,
        neural_network=blocks["neural_network"],
        xgboost=blocks["xgboost"],
        elastic_net=blocks["elastic_net"],
    )


class Stage13NativeStateStore:
    """Content-addressed, tamper-evident native model memoization."""

    def __init__(self, root: Path) -> None:
        if (
            not isinstance(root, Path)
            or not root.is_absolute()
            or root == Path(root.anchor)
        ):
            raise Stage13ProductionRuntimeError(
                "native state store root must be a bounded absolute path"
            )
        self.root = root
        self.objects = root / "objects"

    def _path(self, state_sha256: str) -> Path:
        digest = _require_hash(state_sha256, field="native_state_sha256")
        return self.objects / digest[:2] / f"{digest}.json"

    @staticmethod
    def _require_safe_components(path: Path, *, allow_absent_final: bool) -> None:
        current = Path(path.anchor)
        for index, component in enumerate(path.parts[1:]):
            current = current / component
            final = index == len(path.parts[1:]) - 1
            try:
                metadata = current.stat(follow_symlinks=False)
            except FileNotFoundError:
                if final and allow_absent_final:
                    return
                # An absent parent will be created below; every existing
                # ancestor has already been authenticated as a directory.
                return
            except OSError as exc:
                raise Stage13ProductionRuntimeError(
                    "native state path could not be authenticated"
                ) from exc
            if stat.S_ISLNK(metadata.st_mode):
                raise Stage13ProductionRuntimeError(
                    "native state path contains a symlink"
                )
            if not final and not stat.S_ISDIR(metadata.st_mode):
                raise Stage13ProductionRuntimeError(
                    "native state ancestor is not a directory"
                )
            if final and not allow_absent_final and not stat.S_ISREG(metadata.st_mode):
                raise Stage13ProductionRuntimeError(
                    "native state object is not a regular file"
                )

    def persist(self, state: Stage13NativeFamilyState) -> str:
        if not isinstance(state, Stage13NativeFamilyState):
            raise Stage13ProductionRuntimeError("native state type differs")
        path = self._path(state.state_sha256)
        payload = _canonical_json_bytes(
            {
                "store_schema_version": 1,
                "runtime_id": PRODUCTION_RUNTIME_ID,
                "state": state.payload(include_model_payload=True),
            }
        )
        self._require_safe_components(path, allow_absent_final=True)
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if path.exists():
            observed = _stable_bound_bytes(
                path,
                trusted_root=self.root,
                expected_sha256=_sha256_bytes(payload),
                maximum_size_bytes=max(len(payload), 1),
                label="memoized native state",
            )
            if observed != payload:
                raise Stage13ProductionRuntimeError(
                    "memoized native state content differs"
                )
            return state.state_sha256
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            descriptor = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except OSError as exc:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            raise Stage13ProductionRuntimeError(
                "native state could not be persisted atomically"
            ) from exc
        return state.state_sha256

    def load(self, state_sha256: str) -> Stage13NativeFamilyState:
        path = self._path(state_sha256)
        self._require_safe_components(path, allow_absent_final=False)
        try:
            maximum = path.stat(follow_symlinks=False).st_size
        except OSError as exc:
            raise Stage13ProductionRuntimeError(
                "memoized native state is absent"
            ) from exc
        if maximum <= 0:
            raise Stage13ProductionRuntimeError("memoized native state is empty")
        # The filename is a semantic state digest rather than a raw-file
        # digest. Stable double-read still closes replacement races.
        first = _stable_bound_bytes(
            path,
            trusted_root=self.root,
            expected_sha256=None,
            maximum_size_bytes=maximum,
            label="memoized native state",
        )
        try:
            value = json.loads(first)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise Stage13ProductionRuntimeError(
                "memoized native state is invalid JSON"
            ) from exc
        if (
            not isinstance(value, Mapping)
            or set(value) != {"store_schema_version", "runtime_id", "state"}
            or value.get("store_schema_version") != 1
            or value.get("runtime_id") != PRODUCTION_RUNTIME_ID
            or not isinstance(value.get("state"), Mapping)
        ):
            raise Stage13ProductionRuntimeError(
                "memoized native state envelope differs"
            )
        state = Stage13NativeFamilyState.from_payload(value["state"])
        if state.state_sha256 != state_sha256:
            raise Stage13ProductionRuntimeError("memoized native state address differs")
        return state


class Stage13AdapterStateStore:
    """Content-addressed JSON/Arrow state needed for exact month resume."""

    def __init__(self, root: Path) -> None:
        if (
            not isinstance(root, Path)
            or not root.is_absolute()
            or root == Path(root.anchor)
        ):
            raise Stage13ProductionRuntimeError(
                "adapter state store root must be a bounded absolute path"
            )
        self.root = root
        self.states = root / "states"
        self.frames = root / "frames"

    def _persist_bytes(self, path: Path, payload: bytes, *, label: str) -> None:
        Stage13NativeStateStore._require_safe_components(path, allow_absent_final=True)
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        # Re-authenticate the now-existing parent chain before creating a file.
        Stage13NativeStateStore._require_safe_components(
            path.parent / ".absent", allow_absent_final=True
        )
        if path.exists():
            Stage13NativeStateStore._require_safe_components(
                path, allow_absent_final=False
            )
            observed = _stable_bound_bytes(
                path,
                trusted_root=self.root,
                expected_sha256=_sha256_bytes(payload),
                expected_size_bytes=len(payload),
                label=label,
            )
            if observed != payload:
                raise Stage13ProductionRuntimeError(f"{label} content differs")
            return
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            descriptor = os.open(
                temporary,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            # A hard-link publication is atomic and never overwrites a
            # concurrently completed content-addressed object.
            try:
                os.link(temporary, path, follow_symlinks=False)
            except FileExistsError:
                Stage13NativeStateStore._require_safe_components(
                    path, allow_absent_final=False
                )
                observed = _stable_bound_bytes(
                    path,
                    trusted_root=self.root,
                    expected_sha256=_sha256_bytes(payload),
                    expected_size_bytes=len(payload),
                    label=label,
                )
                if observed != payload:
                    raise Stage13ProductionRuntimeError(
                        f"concurrent {label} content differs"
                    )
            temporary.unlink()
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        except (OSError, Stage13ProductionRuntimeError) as exc:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            if isinstance(exc, Stage13ProductionRuntimeError):
                raise
            raise Stage13ProductionRuntimeError(
                f"{label} could not be persisted atomically"
            ) from exc

    def persist_frame(self, frame: pd.DataFrame) -> str:
        if not isinstance(frame, pd.DataFrame):
            raise Stage13ProductionRuntimeError("memo frame type differs")
        table = pa.Table.from_pandas(
            frame, preserve_index=False
        ).replace_schema_metadata()
        sink = pa.BufferOutputStream()
        with pa_ipc.new_file(sink, table.schema) as writer:
            writer.write_table(table, max_chunksize=65_536)
        payload = sink.getvalue().to_pybytes()
        digest = _sha256_bytes(payload)
        path = self.frames / digest[:2] / f"{digest}.arrow"
        self._persist_bytes(path, payload, label="memoized adapter frame")
        return digest

    def load_frame(self, digest: str) -> pd.DataFrame:
        selected = _require_hash(digest, field="adapter_frame_sha256")
        path = self.frames / selected[:2] / f"{selected}.arrow"
        try:
            size = path.stat(follow_symlinks=False).st_size
        except OSError as exc:
            raise Stage13ProductionRuntimeError(
                "memoized adapter frame is absent"
            ) from exc
        payload = _stable_bound_bytes(
            path,
            trusted_root=self.root,
            expected_sha256=selected,
            expected_size_bytes=size,
            label="memoized adapter frame",
        )
        try:
            return pa_ipc.open_file(pa.BufferReader(payload)).read_all().to_pandas()
        except pa.ArrowException as exc:
            raise Stage13ProductionRuntimeError(
                "memoized adapter frame could not be decoded"
            ) from exc

    def persist_state(self, payload: Mapping[str, object]) -> str:
        normalized = json.loads(_canonical_json_bytes(dict(payload)))
        digest = _sha256_json(normalized)
        envelope = _canonical_json_bytes(
            {
                "store_schema_version": 1,
                "runtime_id": PRODUCTION_RUNTIME_ID,
                "state_sha256": digest,
                "state": normalized,
            }
        )
        path = self.states / digest[:2] / f"{digest}.json"
        self._persist_bytes(path, envelope, label="memoized adapter state")
        return digest

    def load_state(self, digest: str) -> Mapping[str, Any]:
        selected = _require_hash(digest, field="adapter_state_sha256")
        path = self.states / selected[:2] / f"{selected}.json"
        try:
            size = path.stat(follow_symlinks=False).st_size
        except OSError as exc:
            raise Stage13ProductionRuntimeError(
                "memoized adapter state is absent"
            ) from exc
        payload = _stable_bound_bytes(
            path,
            trusted_root=self.root,
            expected_sha256=None,
            expected_size_bytes=size,
            label="memoized adapter state",
        )
        value = _json_mapping(payload, label="memoized adapter state")
        state = value.get("state")
        if (
            set(value)
            != {
                "store_schema_version",
                "runtime_id",
                "state_sha256",
                "state",
            }
            or value.get("store_schema_version") != 1
            or value.get("runtime_id") != PRODUCTION_RUNTIME_ID
            or value.get("state_sha256") != selected
            or not isinstance(state, Mapping)
            or _sha256_json(dict(state)) != selected
        ):
            raise Stage13ProductionRuntimeError(
                "memoized adapter state identity differs"
            )
        return MappingProxyType(dict(state))


@dataclass(frozen=True, slots=True)
class _NativeRiskMemory:
    signal_date: dt.date
    permco: int
    factor_betas: tuple[float, ...]
    specific_variance_monthly: float
    sector: str
    country: str

    def __post_init__(self) -> None:
        if (
            type(self.signal_date) is not dt.date
            or type(self.permco) is not int
            or self.permco <= 0
            or len(self.factor_betas) != len(FACTOR_NAMES)
            or not np.isfinite(np.asarray(self.factor_betas, dtype=float)).all()
            or not np.isfinite(self.specific_variance_monthly)
            or self.specific_variance_monthly <= 0.0
            or not self.sector
            or not self.country
        ):
            raise Stage13ProductionRuntimeError("native risk memory differs")

    def payload(self) -> dict[str, object]:
        return {
            "signal_date": self.signal_date.isoformat(),
            "permco": self.permco,
            "factor_betas_hex": [value.hex() for value in self.factor_betas],
            "specific_variance_monthly_hex": (self.specific_variance_monthly.hex()),
            "sector": self.sector,
            "country": self.country,
        }

    @classmethod
    def from_payload(cls, value: Mapping[str, object]) -> "_NativeRiskMemory":
        expected = {
            "signal_date",
            "permco",
            "factor_betas_hex",
            "specific_variance_monthly_hex",
            "sector",
            "country",
        }
        if set(value) != expected or not isinstance(
            value.get("factor_betas_hex"), list
        ):
            raise Stage13ProductionRuntimeError(
                "persisted native risk memory fields differ"
            )
        try:
            return cls(
                signal_date=dt.date.fromisoformat(str(value["signal_date"])),
                permco=int(value["permco"]),
                factor_betas=tuple(
                    float.fromhex(str(item))
                    for item in value["factor_betas_hex"]  # type: ignore[union-attr]
                ),
                specific_variance_monthly=float.fromhex(
                    str(value["specific_variance_monthly_hex"])
                ),
                sector=str(value["sector"]),
                country=str(value["country"]),
            )
        except (TypeError, ValueError) as exc:
            raise Stage13ProductionRuntimeError(
                "persisted native risk memory differs"
            ) from exc


@dataclass(slots=True)
class _NativeStrategyState:
    portfolio: PortfolioState
    action_ledger: CorporateActionLedger
    mark_ledger: MarkContinuityLedger

    def __post_init__(self) -> None:
        if (
            not isinstance(self.portfolio, PortfolioState)
            or not isinstance(self.action_ledger, CorporateActionLedger)
            or not isinstance(self.mark_ledger, MarkContinuityLedger)
        ):
            raise Stage13ProductionRuntimeError("native strategy state differs")

    def payload(self) -> dict[str, object]:
        return {
            "portfolio": {
                "permno": [int(value) for value in self.portfolio.permno],
                "position_value_dollars_hex": [
                    float(value).hex()
                    for value in self.portfolio.position_value_dollars
                ],
                "cash_dollars_hex": float(self.portfolio.cash_dollars).hex(),
            },
            "action_ledger": {
                "applied_action_ids": list(self.action_ledger.applied_action_ids),
                "last_assigned_return_date": (
                    None
                    if self.action_ledger.last_assigned_return_date is None
                    else self.action_ledger.last_assigned_return_date.isoformat()
                ),
                "last_action_sequence": self.action_ledger.last_action_sequence,
            },
            "mark_ledger": {
                "permno": [int(value) for value in self.mark_ledger.permno],
                "last_authoritative_mark_date": [
                    value.isoformat()
                    for value in self.mark_ledger.last_authoritative_mark_date
                ],
                "pending_no_mark_count": [
                    int(value) for value in self.mark_ledger.pending_no_mark_count
                ],
                "pending_evidence_sha256": list(
                    self.mark_ledger.pending_evidence_sha256
                ),
            },
        }

    @classmethod
    def from_payload(cls, value: Mapping[str, object]) -> "_NativeStrategyState":
        if set(value) != {"portfolio", "action_ledger", "mark_ledger"} or any(
            not isinstance(value.get(name), Mapping)
            for name in ("portfolio", "action_ledger", "mark_ledger")
        ):
            raise Stage13ProductionRuntimeError(
                "persisted native strategy fields differ"
            )
        portfolio = value["portfolio"]  # type: ignore[assignment]
        action = value["action_ledger"]  # type: ignore[assignment]
        mark = value["mark_ledger"]  # type: ignore[assignment]
        try:
            last_action_date = action["last_assigned_return_date"]
            return cls(
                portfolio=PortfolioState(
                    permno=np.asarray(portfolio["permno"], dtype=np.int64),
                    position_value_dollars=np.asarray(
                        [
                            float.fromhex(str(item))
                            for item in portfolio["position_value_dollars_hex"]
                        ],
                        dtype=np.float64,
                    ),
                    cash_dollars=float.fromhex(str(portfolio["cash_dollars_hex"])),
                ),
                action_ledger=CorporateActionLedger(
                    applied_action_ids=tuple(
                        str(item) for item in action["applied_action_ids"]
                    ),
                    last_assigned_return_date=(
                        None
                        if last_action_date is None
                        else dt.date.fromisoformat(str(last_action_date))
                    ),
                    last_action_sequence=int(action["last_action_sequence"]),
                ),
                mark_ledger=MarkContinuityLedger(
                    permno=np.asarray(mark["permno"], dtype=np.int64),
                    last_authoritative_mark_date=tuple(
                        dt.date.fromisoformat(str(item))
                        for item in mark["last_authoritative_mark_date"]
                    ),
                    pending_no_mark_count=np.asarray(
                        mark["pending_no_mark_count"], dtype=np.int64
                    ),
                    pending_evidence_sha256=tuple(
                        str(item) for item in mark["pending_evidence_sha256"]
                    ),
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise Stage13ProductionRuntimeError(
                "persisted native strategy state differs"
            ) from exc


def _calibration_from_payload(
    value: Mapping[str, object],
) -> FinalCalibrationReceipt:
    expected = {
        "feature_scope",
        "strategy_id",
        "fit_signal_date",
        "kappa",
        "unconstrained_slope",
        "observation_count",
        "distinct_month_count",
        "first_holding_month",
        "last_holding_month",
        "pre_final_observation_count",
        "final_prior_live_observation_count",
        "equal_calendar_month_mass",
        "memory_half_life_months",
        "huber_delta",
        "zero_intercept",
        "completed_labels_only",
        "calibration_sha256",
    }
    if (
        set(value) != expected
        or value.get("equal_calendar_month_mass") is not True
        or value.get("zero_intercept") is not True
        or value.get("completed_labels_only") is not True
    ):
        raise Stage13ProductionRuntimeError("persisted final calibration fields differ")
    try:
        return FinalCalibrationReceipt(
            feature_scope=FeatureScope(str(value["feature_scope"])),
            strategy_id=str(value["strategy_id"]),
            fit_signal_date=dt.date.fromisoformat(str(value["fit_signal_date"])),
            kappa=float.fromhex(str(value["kappa"])),
            unconstrained_slope=float.fromhex(str(value["unconstrained_slope"])),
            observation_count=int(value["observation_count"]),
            distinct_month_count=int(value["distinct_month_count"]),
            first_holding_month=str(value["first_holding_month"]),
            last_holding_month=str(value["last_holding_month"]),
            pre_final_observation_count=int(value["pre_final_observation_count"]),
            final_prior_live_observation_count=int(
                value["final_prior_live_observation_count"]
            ),
            completed_labels_only=True,
            calibration_sha256=str(value["calibration_sha256"]),
        )
    except (TypeError, ValueError) as exc:
        raise Stage13ProductionRuntimeError(
            "persisted final calibration differs"
        ) from exc


def _registration_identity_sha256(broker: FinalAccessBroker) -> str:
    identity = getattr(broker, "registration_identity", None)
    if not isinstance(identity, Mapping) or not identity:
        raise Stage13ProductionRuntimeError(
            "verified broker omits its registration pointer/manifest identity"
        )
    return _sha256_json(dict(identity))


@dataclass(frozen=True, slots=True)
class ProductionMonthRequest:
    """One fixed production month, including global predecessor continuity."""

    frozen: FrozenMonthRequest
    approval_payload_sha256: str
    registration_identity_sha256: str
    previous_kernel_state_sha256: str | None
    predecessor_closing_book_sha256_by_strategy: Mapping[str, str] | None
    predecessor_shadow_core_closing_book_sha256_by_strategy: Mapping[str, str] | None

    def __post_init__(self) -> None:
        if not isinstance(self.frozen, FrozenMonthRequest):
            raise Stage13ProductionRuntimeError("frozen month request type differs")
        _require_hash(self.approval_payload_sha256, field="approval_payload_sha256")
        _require_hash(
            self.registration_identity_sha256,
            field="registration_identity_sha256",
        )
        if self.previous_kernel_state_sha256 is not None:
            _require_hash(
                self.previous_kernel_state_sha256,
                field="previous_kernel_state_sha256",
            )
        for field, value in (
            (
                "predecessor_closing_book_sha256_by_strategy",
                self.predecessor_closing_book_sha256_by_strategy,
            ),
            (
                "predecessor_shadow_core_closing_book_sha256_by_strategy",
                self.predecessor_shadow_core_closing_book_sha256_by_strategy,
            ),
        ):
            if value is not None:
                object.__setattr__(
                    self,
                    field,
                    _strict_hash_mapping(
                        value,
                        expected_keys=ACTIVE_STRATEGY_IDS,
                        field=field,
                    ),
                )


@dataclass(frozen=True, slots=True)
class ProductionMonthProduct:
    """All identities and retained rows required to memoize one final month."""

    segment: Stage13Segment
    holding_month: str
    input_sha256_by_registration: Mapping[str, str]
    access_ledger_sha256: str
    registration_identity_sha256: str
    approval_payload_sha256: str
    refit_evidence_sha256: str | None
    forecast_receipt_sha256: str
    economic_kernel_contract_sha256: str
    kernel_previous_state_sha256: str | None
    kernel_state_sha256: str
    adapter_state_sha256: str
    opening_book_sha256_by_strategy: Mapping[str, str]
    closing_book_sha256_by_strategy: Mapping[str, str]
    shadow_core_closing_book_sha256_by_strategy: Mapping[str, str]
    model_state_sha256: Mapping[str, str]
    result_sha256_by_strategy: Mapping[str, str]
    result_fragments: Mapping[str, Mapping[str, Any]]
    runtime_id: str
    product_sha256: str

    def __post_init__(self) -> None:
        plan = next(
            (item for item in FROZEN_SEGMENT_PLANS if item.segment is self.segment),
            None,
        )
        if plan is None or self.holding_month not in plan.holding_months:
            raise Stage13ProductionRuntimeError("production product month differs")
        inputs = _strict_hash_mapping(
            self.input_sha256_by_registration,
            expected_keys=plan.registration_names,
            field="input_sha256_by_registration",
        )
        opening = _strict_hash_mapping(
            self.opening_book_sha256_by_strategy,
            expected_keys=ACTIVE_STRATEGY_IDS,
            field="opening_book_sha256_by_strategy",
        )
        closing = _strict_hash_mapping(
            self.closing_book_sha256_by_strategy,
            expected_keys=ACTIVE_STRATEGY_IDS,
            field="closing_book_sha256_by_strategy",
        )
        shadow_expected = (
            ACTIVE_STRATEGY_IDS if self.segment is Stage13Segment.PRIMARY else ()
        )
        shadow = _strict_hash_mapping(
            self.shadow_core_closing_book_sha256_by_strategy,
            expected_keys=shadow_expected,
            field="shadow_core_closing_book_sha256_by_strategy",
        )
        model = _strict_hash_mapping(
            self.model_state_sha256,
            expected_keys=MODEL_STATE_KEYS,
            field="model_state_sha256",
        )
        result = _strict_hash_mapping(
            self.result_sha256_by_strategy,
            expected_keys=ACTIVE_STRATEGY_IDS,
            field="result_sha256_by_strategy",
        )
        for field in (
            "access_ledger_sha256",
            "registration_identity_sha256",
            "approval_payload_sha256",
            "forecast_receipt_sha256",
            "economic_kernel_contract_sha256",
            "kernel_state_sha256",
            "adapter_state_sha256",
        ):
            _require_hash(getattr(self, field), field=field)
        if (
            self.economic_kernel_contract_sha256
            != FROZEN_FINAL_ECONOMIC_KERNEL_CONTRACT.contract_sha256
        ):
            raise Stage13ProductionRuntimeError(
                "production product economic-kernel contract differs"
            )
        if self.kernel_previous_state_sha256 is not None:
            _require_hash(
                self.kernel_previous_state_sha256,
                field="kernel_previous_state_sha256",
            )
        final_holding_month_descriptor(self.holding_month)
        refit_expected = (
            int(shift_month(self.holding_month, -1)[5:7]) in PRIMARY_REFIT_SIGNAL_MONTHS
        )
        if refit_expected != (self.refit_evidence_sha256 is not None):
            raise Stage13ProductionRuntimeError(
                "quarterly final refit evidence presence differs"
            )
        if self.refit_evidence_sha256 is not None:
            _require_hash(self.refit_evidence_sha256, field="refit_evidence_sha256")
        if not isinstance(self.runtime_id, str) or not self.runtime_id:
            raise Stage13ProductionRuntimeError(
                "production runtime identity is missing"
            )
        fragments = {
            str(name): json.loads(result_json_bytes(value))
            for name, value in self.result_fragments.items()
        }
        if any(name not in STAGE13_RESULT_FILES for name in fragments):
            raise Stage13ProductionRuntimeError(
                "monthly result fragment escaped the closed result set"
            )
        for name, value in (
            ("input_sha256_by_registration", inputs),
            ("opening_book_sha256_by_strategy", opening),
            ("closing_book_sha256_by_strategy", closing),
            ("shadow_core_closing_book_sha256_by_strategy", shadow),
            ("model_state_sha256", model),
            ("result_sha256_by_strategy", result),
            ("result_fragments", MappingProxyType(fragments)),
        ):
            object.__setattr__(self, name, value)
        _require_hash(self.product_sha256, field="product_sha256")
        if self.product_sha256 != _sha256_json(self.payload(False)):
            raise Stage13ProductionRuntimeError(
                "production month product digest differs"
            )

    def payload(self, include_digest: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "segment": self.segment.value,
            "holding_month": self.holding_month,
            "input_sha256_by_registration": dict(self.input_sha256_by_registration),
            "access_ledger_sha256": self.access_ledger_sha256,
            "registration_identity_sha256": self.registration_identity_sha256,
            "approval_payload_sha256": self.approval_payload_sha256,
            "refit_evidence_sha256": self.refit_evidence_sha256,
            "forecast_receipt_sha256": self.forecast_receipt_sha256,
            "economic_kernel_contract_sha256": (self.economic_kernel_contract_sha256),
            "kernel_previous_state_sha256": self.kernel_previous_state_sha256,
            "kernel_state_sha256": self.kernel_state_sha256,
            "adapter_state_sha256": self.adapter_state_sha256,
            "opening_book_sha256_by_strategy": dict(
                self.opening_book_sha256_by_strategy
            ),
            "closing_book_sha256_by_strategy": dict(
                self.closing_book_sha256_by_strategy
            ),
            "shadow_core_closing_book_sha256_by_strategy": dict(
                self.shadow_core_closing_book_sha256_by_strategy
            ),
            "model_state_sha256": dict(self.model_state_sha256),
            "result_sha256_by_strategy": dict(self.result_sha256_by_strategy),
            "result_fragments": dict(self.result_fragments),
            "runtime_id": self.runtime_id,
        }
        if include_digest:
            value["product_sha256"] = self.product_sha256
        return value

    @classmethod
    def create(cls, **values: Any) -> "ProductionMonthProduct":
        normalized = dict(values)
        try:
            normalized["result_fragments"] = json.loads(
                result_json_bytes(values["result_fragments"])
            )
        except (KeyError, Stage13ResultError) as exc:
            raise Stage13ProductionRuntimeError(
                "production month result fragments are not canonical JSON"
            ) from exc
        provisional = cls.__new__(cls)
        for name, value in (*normalized.items(), ("product_sha256", "")):
            object.__setattr__(provisional, name, value)
        digest = _sha256_json(provisional.payload(False))
        return cls(**normalized, product_sha256=digest)

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "ProductionMonthProduct":
        expected = {
            "segment",
            "holding_month",
            "input_sha256_by_registration",
            "access_ledger_sha256",
            "registration_identity_sha256",
            "approval_payload_sha256",
            "refit_evidence_sha256",
            "forecast_receipt_sha256",
            "economic_kernel_contract_sha256",
            "kernel_previous_state_sha256",
            "kernel_state_sha256",
            "adapter_state_sha256",
            "opening_book_sha256_by_strategy",
            "closing_book_sha256_by_strategy",
            "shadow_core_closing_book_sha256_by_strategy",
            "model_state_sha256",
            "result_sha256_by_strategy",
            "result_fragments",
            "runtime_id",
            "product_sha256",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise Stage13ProductionRuntimeError(
                "production month product fields differ"
            )
        try:
            segment = Stage13Segment(str(value["segment"]))
        except ValueError as exc:
            raise Stage13ProductionRuntimeError(
                "production product segment is invalid"
            ) from exc
        mapping_fields = (
            "input_sha256_by_registration",
            "opening_book_sha256_by_strategy",
            "closing_book_sha256_by_strategy",
            "shadow_core_closing_book_sha256_by_strategy",
            "model_state_sha256",
            "result_sha256_by_strategy",
            "result_fragments",
        )
        if any(not isinstance(value[name], Mapping) for name in mapping_fields):
            raise Stage13ProductionRuntimeError(
                "production product nested mapping differs"
            )
        return cls(
            segment=segment,
            holding_month=value["holding_month"],  # type: ignore[arg-type]
            input_sha256_by_registration=value[
                "input_sha256_by_registration"
            ],  # type: ignore[arg-type]
            access_ledger_sha256=value["access_ledger_sha256"],  # type: ignore[arg-type]
            registration_identity_sha256=value[
                "registration_identity_sha256"
            ],  # type: ignore[arg-type]
            approval_payload_sha256=value[
                "approval_payload_sha256"
            ],  # type: ignore[arg-type]
            refit_evidence_sha256=value["refit_evidence_sha256"],  # type: ignore[arg-type]
            forecast_receipt_sha256=value[
                "forecast_receipt_sha256"
            ],  # type: ignore[arg-type]
            economic_kernel_contract_sha256=value[
                "economic_kernel_contract_sha256"
            ],  # type: ignore[arg-type]
            kernel_previous_state_sha256=value[
                "kernel_previous_state_sha256"
            ],  # type: ignore[arg-type]
            kernel_state_sha256=value["kernel_state_sha256"],  # type: ignore[arg-type]
            adapter_state_sha256=value["adapter_state_sha256"],  # type: ignore[arg-type]
            opening_book_sha256_by_strategy=value[
                "opening_book_sha256_by_strategy"
            ],  # type: ignore[arg-type]
            closing_book_sha256_by_strategy=value[
                "closing_book_sha256_by_strategy"
            ],  # type: ignore[arg-type]
            shadow_core_closing_book_sha256_by_strategy=value[
                "shadow_core_closing_book_sha256_by_strategy"
            ],  # type: ignore[arg-type]
            model_state_sha256=value["model_state_sha256"],  # type: ignore[arg-type]
            result_sha256_by_strategy=value[
                "result_sha256_by_strategy"
            ],  # type: ignore[arg-type]
            result_fragments=value["result_fragments"],  # type: ignore[arg-type]
            runtime_id=value["runtime_id"],  # type: ignore[arg-type]
            product_sha256=value["product_sha256"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class CompletedStage13Execution:
    run: Stage13RunResult
    results: Stage13ResultSet
    approval_payload_sha256: str
    registration_identity_sha256: str
    result_set_sha256: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.run, Stage13RunResult)
            or self.run.synthetic
            or not isinstance(self.results, Stage13ResultSet)
            or self.results.synthetic
        ):
            raise Stage13ProductionRuntimeError(
                "completed production execution cannot contain synthetic evidence"
            )
        _require_hash(self.approval_payload_sha256, field="approval_payload_sha256")
        _require_hash(
            self.registration_identity_sha256,
            field="registration_identity_sha256",
        )
        _require_hash(self.result_set_sha256, field="result_set_sha256")
        if self.result_set_sha256 != _sha256_bytes(
            result_json_bytes(self.results.documents)
        ):
            raise Stage13ProductionRuntimeError("result-set digest differs")


class _ProductionAdapter:
    """Sealed internal adapter interface used by the authenticated runner."""

    def restore(self, product: ProductionMonthProduct) -> None:
        raise NotImplementedError

    def execute(self, request: ProductionMonthRequest) -> ProductionMonthProduct:
        raise NotImplementedError

    def finalize(
        self,
        run: Stage13RunResult,
        products: tuple[ProductionMonthProduct, ...],
    ) -> Stage13ResultSet:
        raise NotImplementedError


class _TestProductionAdapter(_ProductionAdapter):
    """Private deterministic seam; never constructed by the public entry."""

    def __init__(
        self,
        *,
        execute: Callable[[ProductionMonthRequest], ProductionMonthProduct],
        finalize: Callable[
            [Stage13RunResult, tuple[ProductionMonthProduct, ...]],
            Stage13ResultSet,
        ],
        _issuance_token: object,
    ) -> None:
        if _issuance_token is not _TEST_ADAPTER_TOKEN:
            raise Stage13ProductionRuntimeError(
                "production test adapter was not privately issued"
            )
        self._execute = execute
        self._finalize = finalize

    def restore(self, product: ProductionMonthProduct) -> None:
        if not isinstance(product, ProductionMonthProduct):
            raise Stage13ProductionRuntimeError("test adapter restore product differs")

    def execute(self, request: ProductionMonthRequest) -> ProductionMonthProduct:
        return self._execute(request)

    def finalize(
        self,
        run: Stage13RunResult,
        products: tuple[ProductionMonthProduct, ...],
    ) -> Stage13ResultSet:
        return self._finalize(run, products)


def _issue_test_production_adapter(
    *,
    execute: Callable[[ProductionMonthRequest], ProductionMonthProduct],
    finalize: Callable[
        [Stage13RunResult, tuple[ProductionMonthProduct, ...]], Stage13ResultSet
    ],
) -> _TestProductionAdapter:
    return _TestProductionAdapter(
        execute=execute,
        finalize=finalize,
        _issuance_token=_TEST_ADAPTER_TOKEN,
    )


def _state_payload(
    request: ProductionMonthRequest,
    product: ProductionMonthProduct,
) -> bytes:
    if (
        product.segment is not request.frozen.plan.segment
        or product.holding_month != request.frozen.holding_month
        or product.approval_payload_sha256 != request.approval_payload_sha256
        or product.registration_identity_sha256 != request.registration_identity_sha256
        or product.kernel_previous_state_sha256 != request.previous_kernel_state_sha256
    ):
        raise Stage13ProductionRuntimeError(
            "production month product does not answer its exact request"
        )
    return _canonical_json_bytes(
        {
            "schema_version": PRODUCTION_STATE_SCHEMA_VERSION,
            "runtime_id": PRODUCTION_RUNTIME_ID,
            "stage12_binding_sha256": request.frozen.stage12_binding_sha256,
            "execution_config_sha256": request.frozen.execution_config_sha256,
            "previous_checkpoint_state_sha256": (request.frozen.previous_state_sha256),
            "approval_payload_sha256": request.approval_payload_sha256,
            "registration_identity_sha256": (request.registration_identity_sha256),
            "product": product.payload(),
        }
    )


def _parse_state(
    loaded: LoadedCheckpoint,
    *,
    plan: FrozenSegmentPlan,
    binding: Stage12VerifiedBinding,
    approval_payload_sha256: str,
    registration_identity_sha256: str,
    execution_config_sha256: str,
) -> ProductionMonthProduct:
    try:
        value = json.loads(loaded.state)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Stage13ProductionRuntimeError(
            "production checkpoint state is invalid JSON"
        ) from exc
    expected = {
        "schema_version",
        "runtime_id",
        "stage12_binding_sha256",
        "execution_config_sha256",
        "previous_checkpoint_state_sha256",
        "approval_payload_sha256",
        "registration_identity_sha256",
        "product",
    }
    if (
        not isinstance(value, Mapping)
        or set(value) != expected
        or value.get("schema_version") != PRODUCTION_STATE_SCHEMA_VERSION
        or value.get("runtime_id") != PRODUCTION_RUNTIME_ID
        or value.get("stage12_binding_sha256") != binding.binding_sha256
        or value.get("execution_config_sha256") != execution_config_sha256
        or value.get("approval_payload_sha256") != approval_payload_sha256
        or value.get("registration_identity_sha256") != registration_identity_sha256
        or not isinstance(value.get("product"), Mapping)
    ):
        raise Stage13ProductionRuntimeError("production checkpoint semantics differ")
    product = ProductionMonthProduct.from_mapping(value["product"])
    if (
        product.segment is not plan.segment
        or product.holding_month != loaded.record.holding_month
    ):
        raise Stage13ProductionRuntimeError(
            "production checkpoint product identity differs"
        )
    return product


def _loaded_chain(
    store: MonthlyCheckpointStore,
    latest: LoadedCheckpoint,
) -> tuple[LoadedCheckpoint, ...]:
    reversed_chain = [latest]
    current = latest.record
    while current.previous_checkpoint_sha256 is not None:
        previous = store._load_record(  # noqa: SLF001 - authenticated store API
            current.previous_checkpoint_sha256,
            load_state=True,
        )
        if not isinstance(previous, LoadedCheckpoint):
            raise Stage13ProductionRuntimeError(
                "production checkpoint ancestry omitted state"
            )
        reversed_chain.append(previous)
        current = previous.record
    return tuple(reversed(reversed_chain))


def _validate_global_successor(
    previous: ProductionMonthProduct | None,
    current: ProductionMonthProduct,
) -> None:
    if previous is None:
        if current.holding_month != "2020-01" or (
            current.kernel_previous_state_sha256 is not None
        ):
            raise Stage13ProductionRuntimeError(
                "primary final chain does not start at its frozen genesis"
            )
        return
    if current.kernel_previous_state_sha256 != previous.kernel_state_sha256:
        raise Stage13ProductionRuntimeError(
            "global economic-kernel predecessor digest differs"
        )
    if previous.holding_month == "2025-08" and current.holding_month == "2025-09":
        if dict(current.opening_book_sha256_by_strategy) != dict(
            previous.shadow_core_closing_book_sha256_by_strategy
        ):
            raise Stage13ProductionRuntimeError(
                "August-to-September shadow-core continuity differs"
            )
        return
    if previous.holding_month == "2025-11" and current.holding_month == "2025-12":
        if dict(current.opening_book_sha256_by_strategy) != dict(
            previous.closing_book_sha256_by_strategy
        ):
            raise Stage13ProductionRuntimeError(
                "November-to-December book continuity differs"
            )
        return
    if previous.segment is not current.segment:
        raise Stage13ProductionRuntimeError("production segment chain was spliced")
    if dict(current.opening_book_sha256_by_strategy) != dict(
        previous.closing_book_sha256_by_strategy
    ):
        raise Stage13ProductionRuntimeError(
            "within-segment inherited book continuity differs"
        )


def _run_authenticated_production(
    *,
    checkpoint_root: Path,
    binding: Stage12VerifiedBinding,
    approval: Stage13ApprovalReceipt,
    registration_identity: Mapping[str, object],
    adapter: _ProductionAdapter,
) -> CompletedStage13Execution:
    """Run or resume the one sealed production state machine."""

    if (
        not isinstance(checkpoint_root, Path)
        or not checkpoint_root.is_absolute()
        or not isinstance(adapter, _ProductionAdapter)
    ):
        raise Stage13ProductionRuntimeError("private production test inputs differ")
    authenticated = require_stage13_unblind_authorization(
        approval, expected_stage12_binding=binding
    )
    registration_sha = _sha256_json(dict(registration_identity))
    execution_sha = frozen_execution_config_sha256()
    all_products: list[ProductionMonthProduct] = []
    segment_results: list[SegmentExecutionResult] = []
    global_previous: ProductionMonthProduct | None = None

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
        loaded_chain = () if latest is None else _loaded_chain(store, latest)
        resumed_products: list[ProductionMonthProduct] = []
        prior_checkpoint_state_sha: str | None = None
        for loaded in loaded_chain:
            product = _parse_state(
                loaded,
                plan=segment_plan,
                binding=binding,
                approval_payload_sha256=authenticated.payload_sha256,
                registration_identity_sha256=registration_sha,
                execution_config_sha256=execution_sha,
            )
            expected_previous = prior_checkpoint_state_sha
            state_value = json.loads(loaded.state)
            if state_value.get("previous_checkpoint_state_sha256") != expected_previous:
                raise Stage13ProductionRuntimeError(
                    "production checkpoint state predecessor differs"
                )
            _validate_global_successor(global_previous, product)
            adapter.restore(product)
            global_previous = product
            resumed_products.append(product)
            all_products.append(product)
            prior_checkpoint_state_sha = loaded.record.state_sha256

        resumed = len(loaded_chain)
        executed: list[str] = []
        for month in segment_plan.holding_months[resumed:]:
            frozen = FrozenMonthRequest(
                plan=segment_plan,
                holding_month=month,
                stage12_binding_sha256=binding.binding_sha256,
                execution_config_sha256=execution_sha,
                previous_state_sha256=prior_checkpoint_state_sha,
            )
            request = ProductionMonthRequest(
                frozen=frozen,
                approval_payload_sha256=authenticated.payload_sha256,
                registration_identity_sha256=registration_sha,
                previous_kernel_state_sha256=(
                    None
                    if global_previous is None
                    else global_previous.kernel_state_sha256
                ),
                predecessor_closing_book_sha256_by_strategy=(
                    None
                    if global_previous is None
                    else global_previous.closing_book_sha256_by_strategy
                ),
                predecessor_shadow_core_closing_book_sha256_by_strategy=(
                    None
                    if global_previous is None
                    or not global_previous.shadow_core_closing_book_sha256_by_strategy
                    else global_previous.shadow_core_closing_book_sha256_by_strategy
                ),
            )
            product = adapter.execute(request)
            _validate_global_successor(global_previous, product)
            latest = store.commit(
                holding_month=month,
                state=_state_payload(request, product),
                state_format=STAGE13_STATE_FORMAT,
            )
            prior_checkpoint_state_sha = latest.record.state_sha256
            global_previous = product
            all_products.append(product)
            executed.append(month)

        if latest is None or latest.record.ordinal != len(segment_plan.holding_months):
            raise Stage13ProductionRuntimeError(
                "production segment did not reach its exact head"
            )
        complete_chain = _loaded_chain(store, latest)
        segment_results.append(
            SegmentExecutionResult(
                plan=segment_plan,
                checkpoint_plan_sha256=checkpoint_plan.plan_sha256,
                head=latest.record,
                chain_records=tuple(item.record for item in complete_chain),
                executed_months=tuple(executed),
                resumed_month_count=resumed,
                synthetic=False,
            )
        )

    run = Stage13RunResult(
        stage12_binding=binding,
        execution_config_sha256=execution_sha,
        segments=tuple(segment_results),
        synthetic=False,
    )
    products = tuple(all_products)
    if len(products) != sum(len(plan.holding_months) for plan in FROZEN_SEGMENT_PLANS):
        raise Stage13ProductionRuntimeError(
            "production month product set is incomplete"
        )
    results = adapter.finalize(run, products)
    if not isinstance(results, Stage13ResultSet) or results.synthetic:
        raise Stage13ProductionRuntimeError(
            "production adapter did not return an empirical Stage13ResultSet"
        )
    return CompletedStage13Execution(
        run=run,
        results=results,
        approval_payload_sha256=authenticated.payload_sha256,
        registration_identity_sha256=registration_sha,
        result_set_sha256=_sha256_bytes(result_json_bytes(results.documents)),
    )


def _run_production_with_test_adapter(
    *,
    checkpoint_root: Path,
    binding: Stage12VerifiedBinding,
    approval: Stage13ApprovalReceipt,
    registration_identity: Mapping[str, object],
    adapter: _TestProductionAdapter,
) -> CompletedStage13Execution:
    """Private synthetic-input harness for the production state machine."""

    if not isinstance(adapter, _TestProductionAdapter):
        raise Stage13ProductionRuntimeError(
            "private production runner requires its sealed test adapter"
        )
    return _run_authenticated_production(
        checkpoint_root=checkpoint_root,
        binding=binding,
        approval=approval,
        registration_identity=registration_identity,
        adapter=adapter,
    )


_SEGMENT_PREFIX = MappingProxyType(
    {
        Stage13Segment.PRIMARY: "primary",
        Stage13Segment.CORE: "core",
        Stage13Segment.PARTIAL: "partial",
    }
)
_REGISTRATION_ROLE = MappingProxyType(
    {
        "final_calendar": FinalPayloadRole.CALENDAR_METADATA,
        **{
            f"{prefix}_predictors": FinalPayloadRole.PREDICTOR
            for prefix in _SEGMENT_PREFIX.values()
        },
        **{
            f"{prefix}_risk_cost": FinalPayloadRole.RISK_INPUT
            for prefix in _SEGMENT_PREFIX.values()
        },
        **{
            f"{prefix}_cost_inputs": FinalPayloadRole.COST_INPUT
            for prefix in _SEGMENT_PREFIX.values()
        },
        **{
            f"{prefix}_factor_covariance": FinalPayloadRole.RISK_INPUT
            for prefix in _SEGMENT_PREFIX.values()
        },
        **{
            f"{prefix}_risk_cost_month_summary": FinalPayloadRole.RISK_INPUT
            for prefix in _SEGMENT_PREFIX.values()
        },
        **{
            f"{prefix}_market_inputs": FinalPayloadRole.MARKET_INPUT
            for prefix in _SEGMENT_PREFIX.values()
        },
        **{
            f"{prefix}_market_accounting": FinalPayloadRole.ACCOUNTING_INPUT
            for prefix in _SEGMENT_PREFIX.values()
        },
        "primary_factors": FinalPayloadRole.FACTOR_RETURN,
        "core_factors": FinalPayloadRole.FACTOR_RETURN,
        "primary_benchmarks": FinalPayloadRole.BENCHMARK_RETURN,
        "core_benchmarks": FinalPayloadRole.BENCHMARK_RETURN,
        "partial_benchmarks": (FinalPayloadRole.PARTIAL_DECEMBER_DESCRIPTIVE_RETURN),
        "primary_completed_labels": FinalPayloadRole.LABEL,
        "core_completed_labels": FinalPayloadRole.LABEL,
        "primary_outcomes": FinalPayloadRole.OUTCOME,
        "core_outcomes": FinalPayloadRole.OUTCOME,
        "partial_descriptive_outcome": (
            FinalPayloadRole.PARTIAL_DECEMBER_DESCRIPTIVE_RETURN
        ),
    }
)
_MODEL_CONFIG_BY_STRATEGY = MappingProxyType(
    {
        PRIMARY_STRATEGY_ID: "nn_001+xgb_core_038_g10",
        ELASTIC_NET_STRATEGY_ID: "enet_r004_l005",
        RANK_STRATEGY_ID: "nn_001+xgb_core_038_g10:rank",
        TRAILING_RESIDUAL_STRATEGY_ID: "trailing_residual_60m_n_over_n_plus_60",
        VQM_STRATEGY_ID: "fixed_value_quality_momentum_v1",
        EQUAL_WEIGHT_STRATEGY_ID: "equal_weight_long_only_v2",
    }
)
_GENESIS_SHA256 = _sha256_json(
    {
        "genesis_rule": "reference_nav_all_cash_then_monthly_primary_chain",
        "reference_nav_dollars": REFERENCE_NAV_DOLLARS,
        "reported_and_shadow_books_are_independent": True,
        "strategy_ids": list(ACTIVE_STRATEGY_IDS),
    }
)


def _table_frame(table: pa.Table, *, date_columns: Sequence[str] = ()) -> pd.DataFrame:
    if not isinstance(table, pa.Table):
        raise Stage13ProductionRuntimeError("authenticated final table type differs")
    frame = table.to_pandas()
    for name in date_columns:
        if name in frame:
            frame[name] = pd.to_datetime(frame[name], errors="raise").dt.date
    return frame


def _portfolio_sha(state: PortfolioState) -> str:
    return PortfolioSnapshot.from_state(state).snapshot_sha256


def _active_scopes(segment: Stage13Segment) -> tuple[FeatureScope, ...]:
    return (
        (
            FeatureScope.FULL_FEATURE_PRIMARY,
            FeatureScope.REDUCED_AUXILIARY_CORE,
        )
        if segment is Stage13Segment.PRIMARY
        else (FeatureScope.REDUCED_AUXILIARY_CORE,)
    )


def _reported_scope(segment: Stage13Segment) -> FeatureScope:
    return (
        FeatureScope.FULL_FEATURE_PRIMARY
        if segment is Stage13Segment.PRIMARY
        else FeatureScope.REDUCED_AUXILIARY_CORE
    )


def _center_feature_with_missing(values: Sequence[object]) -> np.ndarray:
    vector = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(float)
    finite = np.isfinite(vector)
    result = np.zeros(len(vector), dtype=np.float64)
    if finite.any():
        result[finite] = centered_average_rank(vector[finite])
    return result


def _vqm_score_frame(features: pd.DataFrame) -> pd.DataFrame:
    missing = sorted({"signal_date", "permno", *VQM_FEATURES} - set(features))
    if missing:
        raise Stage13ProductionRuntimeError(
            "VQM predictor frame omits frozen fields: " + ", ".join(missing)
        )
    records: list[pd.DataFrame] = []
    for signal_date, group in features.groupby("signal_date", sort=True, observed=True):
        ordered = group.sort_values("permno", kind="mergesort").reset_index(drop=True)
        ranked = {
            name: _center_feature_with_missing(ordered[name]) for name in VQM_FEATURES
        }
        score = value_quality_momentum_scores(ranked).centered_score
        records.append(
            pd.DataFrame(
                {
                    "signal_date": [signal_date] * len(ordered),
                    "permno": ordered["permno"].to_numpy(np.int64),
                    "vqm_score": score,
                }
            )
        )
    if not records:
        raise Stage13ProductionRuntimeError("VQM predictor history is empty")
    return (
        pd.concat(records, ignore_index=True)
        .sort_values(["signal_date", "permno"], kind="mergesort")
        .reset_index(drop=True)
    )


def _monthly_identity(
    *,
    signal_date: dt.date,
    holding_month: str,
    decision_inputs: Mapping[str, str],
    calendar_access_receipt_sha256: str,
) -> AuthenticatedMonthlyIdentity:
    return AuthenticatedMonthlyIdentity(
        signal_date=signal_date,
        source_bundle_sha256=_sha256_json(
            {
                "stage": 13,
                "holding_month": holding_month,
                "decision_input_sha256": dict(sorted(decision_inputs.items())),
                "calendar_access_receipt_sha256": (calendar_access_receipt_sha256),
                "accounting_or_outcome_rows_opened": False,
            }
        ),
    )


def _monthly_market_and_risk(
    *,
    identity: AuthenticatedMonthlyIdentity,
    risk_frame: pd.DataFrame,
    market_frame: pd.DataFrame,
    covariance_frame: pd.DataFrame,
    summary_frame: pd.DataFrame,
    inherited_identity: Mapping[int, _NativeRiskMemory] | None = None,
) -> tuple[pd.DataFrame, MarketSnapshot, MonthlyRiskInputs]:
    risk_audit = risk_frame.copy()
    decision_market = market_frame.copy()
    if (
        risk_audit.empty
        or risk_audit["permno"].duplicated().any()
        or decision_market.empty
        or decision_market["permno"].duplicated().any()
    ):
        raise Stage13ProductionRuntimeError(
            "monthly final risk/market domain is empty or duplicated"
        )
    for frame in (risk_audit, decision_market):
        frame["permno"] = pd.to_numeric(frame["permno"], errors="raise").astype(
            np.int64
        )
    if "long_eligible" not in risk_audit:
        raise Stage13ProductionRuntimeError(
            "monthly final risk audit omits long eligibility"
        )
    # Stage 8 publishes the complete predictor audit spine, whereas the
    # decision-side market authority contains current or historically
    # long-eligible names.  Never-admitted audit rows are not decision assets.
    current_risk = risk_audit.loc[risk_audit["long_eligible"].astype(bool)].copy()
    if current_risk.empty:
        raise Stage13ProductionRuntimeError(
            "monthly final current risk decision domain is empty"
        )
    risk_fields = (
        "permno",
        "permco",
        "long_eligible",
        "short_eligible",
        "ff12_sector_category",
        "domicile_country_category",
        "risk_status",
        *(f"beta_risk_{name}" for name in FACTOR_NAMES),
        "specific_variance_monthly",
        "full_spread_20d",
        "adv20_usd",
        "impact_volatility_daily",
        "new_long_trade_allowed",
        "new_short_trade_allowed",
    )
    market_fields = (
        "permno",
        "permco",
        "long_eligible",
        "short_eligible",
        "close_usd_per_share",
        "entry_close_status",
        "full_spread_20d",
        "adv20_usd",
        "impact_volatility_daily",
        "new_long_trade_allowed",
        "new_short_trade_allowed",
    )
    for label, frame, fields in (
        ("risk", current_risk, risk_fields),
        ("market", decision_market, market_fields),
    ):
        missing = sorted(set(fields) - set(frame))
        if missing:
            raise Stage13ProductionRuntimeError(
                f"monthly final {label} rows omit fields: " + ", ".join(missing)
            )
    remembered = {} if inherited_identity is None else dict(inherited_identity)
    if any(
        type(permno) is not int
        or permno <= 0
        or not isinstance(memory, _NativeRiskMemory)
        for permno, memory in remembered.items()
    ):
        raise Stage13ProductionRuntimeError(
            "monthly inherited market identity authority differs"
        )
    current_risk_ids = set(int(value) for value in current_risk["permno"])
    physical_market_ids = set(int(value) for value in decision_market["permno"])
    physical_market_row_count = len(decision_market)
    inherited_only_ids = sorted(
        set(remembered) - current_risk_ids - physical_market_ids
    )
    if inherited_only_ids:
        inherited_only_rows = [
            {
                "permno": permno,
                "permco": remembered[permno].permco,
                "long_eligible": False,
                "short_eligible": False,
                "close_usd_per_share": np.nan,
                "entry_close_status": "MISSING_VALID_ENTRY_CLOSE",
                "full_spread_20d": np.nan,
                "adv20_usd": np.nan,
                "impact_volatility_daily": np.nan,
                "new_long_trade_allowed": False,
                "new_short_trade_allowed": False,
            }
            for permno in inherited_only_ids
        ]
        decision_market = pd.concat(
            (decision_market, pd.DataFrame(inherited_only_rows)),
            ignore_index=True,
        )
    market_columns = decision_market.loc[:, list(market_fields)].rename(
        columns={name: f"market_{name}" for name in market_fields if name != "permno"}
    )
    audit = market_columns.merge(
        current_risk.loc[:, list(risk_fields)],
        on="permno",
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    both = audit["_merge"].eq("both").to_numpy(bool)
    market_only = audit["_merge"].eq("left_only").to_numpy(bool)
    risk_only = audit["_merge"].eq("right_only").to_numpy(bool)
    current_risk_audit = both | risk_only
    inherited_only = audit["permno"].isin(inherited_only_ids).to_numpy(bool)
    authenticated_entry_execution = (both | market_only) & ~inherited_only
    if (
        int(np.sum(current_risk_audit)) != len(current_risk)
        or int(np.sum(authenticated_entry_execution)) != physical_market_row_count
    ):
        raise Stage13ProductionRuntimeError(
            "monthly final current-risk/market identity differs"
        )
    for name in ("permco", "long_eligible", "short_eligible"):
        left = audit.loc[both, f"market_{name}"].to_numpy()
        right = audit.loc[both, name].to_numpy()
        if not np.array_equal(left, right):
            raise Stage13ProductionRuntimeError(
                f"monthly final risk/market {name} differs"
            )
    for name in (
        "full_spread_20d",
        "adv20_usd",
        "impact_volatility_daily",
    ):
        left = pd.to_numeric(
            audit.loc[both, f"market_{name}"], errors="coerce"
        ).to_numpy(float)
        right = pd.to_numeric(audit.loc[both, name], errors="coerce").to_numpy(float)
        if not np.array_equal(left, right, equal_nan=True):
            raise Stage13ProductionRuntimeError(
                f"monthly final risk/market {name} differs"
            )
    for name in ("new_long_trade_allowed", "new_short_trade_allowed"):
        if not np.array_equal(
            audit.loc[both, f"market_{name}"].to_numpy(bool),
            audit.loc[both, name].to_numpy(bool),
        ):
            raise Stage13ProductionRuntimeError(
                f"monthly final risk/market {name} differs"
            )

    # The frozen entry-market source uses an authenticated inner join and
    # therefore omits closed delisting tombstones that remain in the current
    # risk/scoring spine.  Preserve those rows as explicit zero-capacity
    # decision identities: absence proves that no entry execution is
    # authenticated, but it must not erase their score or current risk record.
    if np.any(risk_only):
        for name in ("permco", "long_eligible", "short_eligible"):
            audit.loc[risk_only, f"market_{name}"] = audit.loc[risk_only, name]
        audit.loc[risk_only, "market_close_usd_per_share"] = np.nan
        audit.loc[risk_only, "market_entry_close_status"] = "MISSING_VALID_ENTRY_CLOSE"
        for name in (
            "full_spread_20d",
            "adv20_usd",
            "impact_volatility_daily",
        ):
            audit.loc[risk_only, f"market_{name}"] = audit.loc[risk_only, name]
        audit.loc[risk_only, "market_new_long_trade_allowed"] = audit.loc[
            risk_only, "new_long_trade_allowed"
        ]
        audit.loc[risk_only, "market_new_short_trade_allowed"] = audit.loc[
            risk_only, "new_short_trade_allowed"
        ]

    audit.loc[market_only, "permco"] = audit.loc[market_only, "market_permco"]
    # A market-only name is authenticated solely to liquidate an inherited
    # position.  A stale prior predictor/risk row must never reopen new risk.
    audit.loc[market_only, "long_eligible"] = False
    audit.loc[market_only, "short_eligible"] = False
    audit.loc[market_only, "ff12_sector_category"] = None
    audit.loc[market_only, "domicile_country_category"] = None
    audit.loc[market_only, "risk_status"] = "MISSING_CURRENT_RISK"
    for name in (
        *(f"beta_risk_{factor}" for factor in FACTOR_NAMES),
        "specific_variance_monthly",
    ):
        audit.loc[market_only, name] = np.nan
    for name in (
        "full_spread_20d",
        "adv20_usd",
        "impact_volatility_daily",
    ):
        audit[name] = audit[f"market_{name}"]
    audit.loc[market_only, "new_long_trade_allowed"] = False
    audit.loc[market_only, "new_short_trade_allowed"] = False
    audit["close_usd_per_share"] = audit["market_close_usd_per_share"]
    audit["entry_close_status"] = audit["market_entry_close_status"]
    audit["permco"] = pd.to_numeric(audit["permco"], errors="raise").astype(np.int64)
    audit["long_eligible"] = audit["long_eligible"].astype(bool)
    audit["short_eligible"] = audit["short_eligible"].astype(bool)
    audit["new_long_trade_allowed"] = audit["new_long_trade_allowed"].astype(bool)
    audit["new_short_trade_allowed"] = audit["new_short_trade_allowed"].astype(bool)
    audit["current_risk_audit"] = current_risk_audit
    audit["authenticated_entry_execution"] = authenticated_entry_execution
    audit = audit.drop(
        columns=[name for name in audit if name.startswith("market_")] + ["_merge"]
    )
    if audit["entry_close_status"].isna().any():
        raise Stage13ProductionRuntimeError(
            "monthly final market row lacks authenticated entry execution status"
        )
    audit = audit.sort_values("permno", kind="mergesort").reset_index(drop=True)
    numeric = lambda name: pd.to_numeric(audit[name], errors="coerce").to_numpy(float)
    market = MarketSnapshot(
        identity=identity,
        permno=audit["permno"].to_numpy(np.int64),
        permco=pd.to_numeric(audit["permco"], errors="raise").to_numpy(np.int64),
        execution_price=numeric("close_usd_per_share"),
        full_spread=numeric("full_spread_20d"),
        adv20_usd=numeric("adv20_usd"),
        impact_volatility_daily=numeric("impact_volatility_daily"),
        factor_betas=np.column_stack(
            [numeric(f"beta_risk_{name}") for name in FACTOR_NAMES]
        ),
        specific_variance_monthly=numeric("specific_variance_monthly"),
        sector=audit["ff12_sector_category"]
        .fillna("UNKNOWN_SECTOR")
        .astype(str)
        .to_numpy(),
        country=audit["domicile_country_category"]
        .fillna("UNKNOWN_COUNTRY")
        .astype(str)
        .to_numpy(),
        new_long_trade_allowed=audit["new_long_trade_allowed"].astype(bool).to_numpy(),
        new_short_trade_allowed=audit["new_short_trade_allowed"]
        .astype(bool)
        .to_numpy(),
    )
    covariance = covariance_frame.sort_values(
        ["factor_row_ordinal", "factor_col_ordinal"], kind="mergesort"
    ).reset_index(drop=True)
    if len(covariance) != 36:
        raise Stage13ProductionRuntimeError(
            "monthly final factor covariance is not 6 by 6"
        )
    row_ord = pd.to_numeric(covariance["factor_row_ordinal"], errors="raise").to_numpy(
        int
    )
    col_ord = pd.to_numeric(covariance["factor_col_ordinal"], errors="raise").to_numpy(
        int
    )
    matrix = (
        pd.to_numeric(covariance["covariance_monthly"], errors="coerce")
        .to_numpy(float)
        .reshape(6, 6)
    )
    if (
        not np.array_equal(row_ord, np.repeat(np.arange(6), 6))
        or not np.array_equal(col_ord, np.tile(np.arange(6), 6))
        or tuple(covariance["factor_row"].iloc[::6].astype(str)) != FACTOR_NAMES
        or tuple(covariance["factor_col"].iloc[:6].astype(str)) != FACTOR_NAMES
        or not np.isfinite(matrix).all()
        or not np.allclose(matrix, matrix.T, rtol=0.0, atol=1e-12)
        or float(np.min(np.linalg.eigvalsh(matrix))) < -1e-10
    ):
        raise Stage13ProductionRuntimeError(
            "monthly final factor covariance identity differs"
        )
    present = audit.loc[
        audit["risk_status"].astype(str).eq(PRESENT)
        & audit["current_risk_audit"].astype(bool),
        "specific_variance_monthly",
    ]
    values = pd.to_numeric(present, errors="coerce").to_numpy(float)
    values = values[np.isfinite(values) & (values > 0.0)]
    p95 = type7_quantile(values, 0.95)
    if p95 is None or p95 <= 0.0 or len(summary_frame) != 1:
        raise Stage13ProductionRuntimeError(
            "monthly final risk fallback authority differs"
        )
    risk = MonthlyRiskInputs(
        identity=identity,
        factor_covariance_monthly=matrix,
        specific_variance_p95_monthly=float(p95),
        required_exit_impact_volatility_p95_daily=float(
            summary_frame.iloc[0]["exit_volatility_p95_daily"]
        ),
    )
    return audit, market, risk


def _apply_retired_identity_execution_gate(
    market: MarketSnapshot,
    *,
    retired_on: Mapping[int, dt.date],
    entry_date: dt.date,
) -> MarketSnapshot:
    """Port the frozen Stage 11 terminal-identity entry gate unchanged.

    A resolved terminal action processed in an already completed holding
    period retires its source PERMNO for every later decision.  The identity,
    score-domain membership, execution observations, and risk observations
    remain untouched; only new long and short entry authority is closed.
    """

    if (
        not isinstance(market, MarketSnapshot)
        or not isinstance(retired_on, Mapping)
        or type(entry_date) is not dt.date
    ):
        raise Stage13ProductionRuntimeError(
            "retired-identity execution gate inputs differ"
        )
    normalized: dict[int, dt.date] = {}
    for raw_permno, retirement_date in retired_on.items():
        if (
            type(raw_permno) is not int
            or raw_permno <= 0
            or type(retirement_date) is not dt.date
            or retirement_date > entry_date
        ):
            raise Stage13ProductionRuntimeError(
                "retired-identity execution gate received a future or invalid event"
            )
        normalized[raw_permno] = retirement_date

    long_allowed = market.new_long_trade_allowed.copy()
    short_allowed = market.new_short_trade_allowed.copy()
    for position, raw_permno in enumerate(market.permno):
        if int(raw_permno) in normalized:
            long_allowed[position] = False
            short_allowed[position] = False
    return replace(
        market,
        new_long_trade_allowed=long_allowed,
        new_short_trade_allowed=short_allowed,
    )


def _advance_retired_identity_state(
    retired_on: Mapping[int, dt.date],
    *,
    actions: Sequence[CorporateAction],
    entry_date: dt.date,
    exit_date: dt.date,
) -> dict[int, dt.date]:
    """Atomically add this month's resolved terminal source identities."""

    if (
        not isinstance(retired_on, Mapping)
        or isinstance(actions, (str, bytes))
        or not isinstance(actions, Sequence)
        or type(entry_date) is not dt.date
        or type(exit_date) is not dt.date
        or entry_date >= exit_date
    ):
        raise Stage13ProductionRuntimeError(
            "retired-identity state transition inputs differ"
        )
    result: dict[int, dt.date] = {}
    for raw_permno, retirement_date in retired_on.items():
        if (
            type(raw_permno) is not int
            or raw_permno <= 0
            or type(retirement_date) is not dt.date
            or retirement_date > entry_date
        ):
            raise Stage13ProductionRuntimeError(
                "retired-identity state contains a future or invalid event"
            )
        result[raw_permno] = retirement_date
    for action in actions:
        if (
            not isinstance(action, CorporateAction)
            or not entry_date < action.assigned_return_date <= exit_date
        ):
            raise Stage13ProductionRuntimeError(
                "retired-identity action escaped the authenticated holding period"
            )
        prior = result.get(action.source_permno)
        if prior is not None and prior != action.assigned_return_date:
            raise Stage13ProductionRuntimeError(
                "terminal source PERMNO was retired on conflicting dates"
            )
        result[action.source_permno] = action.assigned_return_date
    return result


def _retired_identity_state_payload(
    retired_on: Mapping[int, dt.date],
) -> dict[str, str]:
    """Serialize the shared retired-identity continuation state canonically."""

    if not isinstance(retired_on, Mapping):
        raise Stage13ProductionRuntimeError("native retired-identity state differs")
    items = tuple(retired_on.items())
    if any(
        type(raw_permno) is not int
        or raw_permno <= 0
        or type(retirement_date) is not dt.date
        for raw_permno, retirement_date in items
    ):
        raise Stage13ProductionRuntimeError("native retired-identity state differs")
    result: dict[str, str] = {}
    for raw_permno, retirement_date in sorted(items):
        result[str(raw_permno)] = retirement_date.isoformat()
    return result


def _retired_identity_state_from_payload(
    value: object,
    *,
    completed_session_dates: Sequence[dt.date],
) -> dict[int, dt.date]:
    """Restore and bind retirement dates to already completed sessions."""

    if not isinstance(value, Mapping):
        raise Stage13ProductionRuntimeError("persisted retired-identity state differs")
    sessions = tuple(completed_session_dates)
    if (
        isinstance(completed_session_dates, (str, bytes))
        or not isinstance(completed_session_dates, Sequence)
        or any(type(item) is not dt.date for item in sessions)
        or sessions != tuple(sorted(set(sessions)))
    ):
        raise Stage13ProductionRuntimeError(
            "persisted completed-session history differs"
        )
    session_set = set(sessions)
    result: dict[int, dt.date] = {}
    try:
        for raw_permno, raw_date in value.items():
            if type(raw_permno) is not str or not re.fullmatch(
                r"[1-9][0-9]*", raw_permno
            ):
                raise ValueError("noncanonical retired PERMNO")
            permno = int(raw_permno)
            retirement_date = dt.date.fromisoformat(str(raw_date))
            if retirement_date not in session_set or permno in result:
                raise ValueError("retired event is outside completed sessions")
            result[permno] = retirement_date
    except (TypeError, ValueError) as exc:
        raise Stage13ProductionRuntimeError(
            "persisted retired-identity state differs"
        ) from exc
    return result


def _update_native_risk_memory(
    memory: dict[int, _NativeRiskMemory],
    audit: pd.DataFrame,
    signal_date: dt.date,
) -> None:
    for row in audit.loc[audit["risk_status"].astype(str).eq(PRESENT)].itertuples(
        index=False
    ):
        betas = tuple(float(getattr(row, f"beta_risk_{name}")) for name in FACTOR_NAMES)
        memory[int(row.permno)] = _NativeRiskMemory(
            signal_date=signal_date,
            permco=int(row.permco),
            factor_betas=betas,
            specific_variance_monthly=float(row.specific_variance_monthly),
            sector=(
                "UNKNOWN_SECTOR"
                if pd.isna(row.ff12_sector_category)
                else str(row.ff12_sector_category)
            ),
            country=(
                "UNKNOWN_COUNTRY"
                if pd.isna(row.domicile_country_category)
                else str(row.domicile_country_category)
            ),
        )


def _native_inherited_holdings(
    *,
    state: PortfolioState,
    identity: AuthenticatedMonthlyIdentity,
    audit: pd.DataFrame,
    memory: Mapping[int, _NativeRiskMemory],
    session_dates: Sequence[dt.date],
) -> InheritedHoldings:
    nonzero = np.flatnonzero(state.position_value_dollars != 0.0)
    if nonzero.size == 0:
        return InheritedHoldings.empty(identity)
    by_permno = {int(row.permno): row for row in audit.itertuples(index=False)}
    records: list[dict[str, object]] = []
    for position in nonzero:
        permno = int(state.permno[position])
        remembered = memory.get(permno)
        if remembered is None:
            raise Stage13ProductionRuntimeError(
                f"inherited final holding lacks risk memory: {permno}"
            )
        current = by_permno.get(permno)
        current_long = bool(current.long_eligible) if current is not None else False
        current_short = bool(current.short_eligible) if current is not None else False
        weight = float(state.weights[position])
        records.append(
            {
                "permno": permno,
                "permco": (
                    int(current.permco) if current is not None else remembered.permco
                ),
                "previous_weight": weight,
                "factor_betas": remembered.factor_betas,
                "specific": remembered.specific_variance_monthly,
                "age": sum(
                    remembered.signal_date < value <= identity.signal_date
                    for value in session_dates
                ),
                "sector": (
                    remembered.sector
                    if current is None or pd.isna(current.ff12_sector_category)
                    else str(current.ff12_sector_category)
                ),
                "country": (
                    remembered.country
                    if current is None or pd.isna(current.domicile_country_category)
                    else str(current.domicile_country_category)
                ),
                "current_long": current_long,
                "current_short": current_short,
                "required_exit": (weight > 0.0 and not current_long)
                or (weight < 0.0 and not current_short),
            }
        )
    records.sort(key=lambda item: int(item["permno"]))
    return InheritedHoldings(
        identity=identity,
        permno=np.asarray([item["permno"] for item in records], dtype=np.int64),
        permco=np.asarray([item["permco"] for item in records], dtype=np.int64),
        previous_weight=np.asarray(
            [item["previous_weight"] for item in records], dtype=np.float64
        ),
        last_factor_betas=np.asarray(
            [item["factor_betas"] for item in records], dtype=np.float64
        ),
        last_specific_variance_monthly=np.asarray(
            [item["specific"] for item in records], dtype=np.float64
        ),
        risk_age_sessions=np.asarray([item["age"] for item in records], dtype=np.int64),
        sector=np.asarray([item["sector"] for item in records], dtype=str),
        country=np.asarray([item["country"] for item in records], dtype=str),
        current_long_eligible=np.asarray(
            [item["current_long"] for item in records], dtype=bool
        ),
        current_short_eligible=np.asarray(
            [item["current_short"] for item in records], dtype=bool
        ),
        required_exit=np.asarray(
            [item["required_exit"] for item in records], dtype=bool
        ),
    )


def _scored_snapshot(
    *,
    identity: AuthenticatedMonthlyIdentity,
    permno: np.ndarray,
    score: np.ndarray,
    predictor_frame: pd.DataFrame,
    kappa: float,
    admitted: np.ndarray | None = None,
) -> ScoredSnapshot:
    eligibility = predictor_frame.loc[
        :, ["permno", "long_eligible", "short_eligible"]
    ].copy()
    eligibility["permno"] = pd.to_numeric(eligibility["permno"], errors="raise").astype(
        np.int64
    )
    if eligibility["permno"].duplicated().any():
        raise Stage13ProductionRuntimeError(
            "final predictor eligibility domain is duplicated"
        )
    keyed = eligibility.set_index("permno")
    try:
        ordered = keyed.loc[[int(value) for value in permno]]
    except KeyError as exc:
        raise Stage13ProductionRuntimeError(
            "final native score escaped predictor eligibility"
        ) from exc
    long = ordered["long_eligible"].astype(bool).to_numpy()
    short = ordered["short_eligible"].astype(bool).to_numpy()
    if admitted is not None:
        available = np.asarray(admitted, dtype=bool)
        if available.shape != long.shape:
            raise Stage13ProductionRuntimeError("strategy score admission mask differs")
        long &= available
        short &= available
    values = np.asarray(score, dtype=np.float64).copy()
    if values.shape != long.shape:
        raise Stage13ProductionRuntimeError("strategy score shape differs")
    values[~long] = np.nan
    return ScoredSnapshot(
        identity=identity,
        permno=np.asarray(permno, dtype=np.int64),
        score=values,
        calibration_kappa_monthly=float(kappa),
        long_eligible=long,
        short_eligible=short,
    )


def _completed_residual_expected_returns(
    *,
    permno: np.ndarray,
    labels: pd.DataFrame,
    fit_signal_date: dt.date,
    fit_holding_month: str,
) -> tuple[np.ndarray, np.ndarray, Mapping[int, float], str]:
    # Preserve the frozen 60 *calendar-month slot* window before filtering
    # unavailable observations.  Filtering to present rows first would silently
    # backfill a missing in-window month with an older return.
    window_months = tuple(
        shift_month(fit_holding_month, offset) for offset in range(-61, -1)
    )
    history = labels.loc[labels["holding_month"].astype(str).isin(window_months)].copy()
    history["permno"] = pd.to_numeric(history["permno"], errors="raise").astype(
        np.int64
    )
    if history.duplicated(["holding_month", "permno"]).any():
        raise Stage13ProductionRuntimeError(
            "trailing-residual history has duplicate security-month rows"
        )
    history["_residual"] = pd.to_numeric(
        history["unranked_residual_return"], errors="coerce"
    )
    usable = (
        history["primary_label_status"].astype(str).eq(PRESENT)
        & history["_residual"].notna()
        & history["exit_date"].le(fit_signal_date)
        & history["outcome_availability_date"].le(fit_signal_date)
        & history["holding_month"].astype(str).lt(fit_holding_month)
    )
    history.loc[~usable, "_residual"] = np.nan
    history = history.sort_values(["holding_month", "permno"], kind="mergesort")
    estimates = np.full(len(permno), np.nan, dtype=np.float64)
    for position, raw_permno in enumerate(permno):
        security_history = history.loc[
            history["permno"].eq(int(raw_permno)),
            ["holding_month", "_residual"],
        ].set_index("holding_month")
        values = security_history["_residual"].reindex(window_months).to_numpy()
        estimate = trailing_residual_mean_estimate(values)
        if estimate.available:
            estimates[position] = float(estimate.expected_return)
    available = np.isfinite(estimates)
    score = np.full(len(estimates), np.nan, dtype=np.float64)
    if available.any():
        score[available] = centered_average_rank(estimates[available])
    expected = MappingProxyType(
        {
            int(permno[position]): float(estimates[position])
            for position in np.flatnonzero(available)
        }
    )
    usable_history = history.loc[history["_residual"].notna()]
    training_cutoff = (
        shift_month(fit_holding_month, -2)
        if usable_history.empty
        else str(usable_history["holding_month"].max())
    )
    if training_cutoff >= fit_holding_month:
        raise Stage13ProductionRuntimeError(
            "trailing-residual training cutoff crossed the decision month"
        )
    return score, available, expected, training_cutoff


def _monthly_ic(
    rows: Sequence[Mapping[str, object]],
    *,
    labels: pd.DataFrame | None,
    outcomes: pd.DataFrame,
    partial: bool,
) -> float:
    score = {int(row["permno"]): float(row["score"]) for row in rows}
    if partial:
        outcome_name = "partial_total_return"
        target = outcomes.loc[:, ["permno", outcome_name]].copy()
    elif labels is not None:
        target = labels.loc[
            labels["primary_label_status"].astype(str).eq(PRESENT),
            ["permno", "residual_return_rank"],
        ].copy()
        outcome_name = "residual_return_rank"
    else:  # pragma: no cover - complete callers always retain labels
        target = outcomes.loc[:, ["permno", "total_return"]].copy()
        outcome_name = "total_return"
    pairs = [
        (score[int(row.permno)], float(getattr(row, outcome_name)))
        for row in target.itertuples(index=False)
        if int(row.permno) in score
        and math.isfinite(score[int(row.permno)])
        and math.isfinite(float(getattr(row, outcome_name)))
    ]
    if len(pairs) < 2:
        return 0.0
    result = correlation_with_status(
        [value[0] for value in pairs],
        [value[1] for value in pairs],
        method="spearman",
    )
    return float(result.value) if math.isfinite(result.value) else 0.0


def _model_fragment_rows(
    *,
    segment: Stage13Segment,
    holding_month: str,
    scored_by_strategy: Mapping[str, ScoredSnapshot],
    forecast: FinalForecastReceipt,
    direct_expected_return_by_permno: Mapping[int, float],
    trailing_training_cutoff_month: str,
    calibrations: Mapping[str, FinalCalibrationReceipt],
    model_states: Mapping[str, Stage13NativeFamilyState],
    labels: pd.DataFrame | None,
    outcomes: pd.DataFrame,
    checkpoint_sha256: str,
) -> list[dict[str, object]]:
    scope = _reported_scope(segment)
    if (
        set(scored_by_strategy) != set(ACTIVE_STRATEGY_IDS)
        or forecast.feature_scope is not scope
        or forecast.holding_month != holding_month
    ):
        raise Stage13ProductionRuntimeError(
            "model-evidence scored/forecast identity differs"
        )
    forecast_index = {
        int(permno): position for position, permno in enumerate(forecast.permno)
    }
    if len(forecast_index) != len(forecast.permno):
        raise Stage13ProductionRuntimeError(
            "model-evidence forecast universe is duplicated"
        )
    native_by_family: dict[str, Stage13NativeFamilyState] = {}
    for family in ("neural_network", "xgboost", "elastic_net"):
        key = f"{scope.value}:{family}"
        try:
            native_by_family[family] = model_states[key]
        except KeyError as exc:
            raise Stage13ProductionRuntimeError(
                f"model-evidence lacks active {scope.value} {family} state"
            ) from exc

    def native_identity(
        families: Sequence[str],
    ) -> tuple[str, dt.date]:
        selected = {family: native_by_family[family] for family in families}
        dates = {
            value.full_preprocessor.refit_signal_date for value in selected.values()
        }
        if len(dates) != 1:
            raise Stage13ProductionRuntimeError(
                "strategy model families do not share one active refit date"
            )
        return (
            _sha256_json(
                {
                    "feature_scope": scope.value,
                    "native_state_sha256_by_family": {
                        family: value.state_sha256
                        for family, value in sorted(selected.items())
                    },
                }
            ),
            next(iter(dates)),
        )

    records: list[dict[str, object]] = []
    for strategy_id in ACTIVE_STRATEGY_IDS:
        scored = scored_by_strategy[strategy_id]
        if scored.identity.signal_date.strftime("%Y-%m") != shift_month(
            holding_month, -1
        ):
            raise Stage13ProductionRuntimeError(
                "model-evidence scoring chronology differs"
            )
        admitted = np.asarray(scored.long_eligible, dtype=bool) & np.isfinite(
            np.asarray(scored.score, dtype=float)
        )
        if not admitted.any():
            raise Stage13ProductionRuntimeError(
                f"model-evidence admitted universe is empty: {strategy_id}"
            )
        calibration = calibrations.get(f"{scope.value}:{strategy_id}")
        if (strategy_id in CALIBRATED_STRATEGY_IDS) != (calibration is not None):
            raise Stage13ProductionRuntimeError(
                "model-evidence calibration state differs from strategy contract"
            )
        slope = None if calibration is None else float(calibration.kappa)
        model_config = _MODEL_CONFIG_BY_STRATEGY[strategy_id]
        if strategy_id in {PRIMARY_STRATEGY_ID, RANK_STRATEGY_ID}:
            native_sha, refit_signal_date = native_identity(
                ("neural_network", "xgboost")
            )
            if (
                strategy_id == PRIMARY_STRATEGY_ID
                and calibration is not None
                and calibration.fit_signal_date != refit_signal_date
            ):
                raise Stage13ProductionRuntimeError(
                    "primary model and shared calibration refit dates differ"
                )
            refit_id = (
                native_sha
                if calibration is None
                else _sha256_json(
                    {
                        "native_state_sha256": native_sha,
                        "shared_calibration_policy_id": (SHARED_CALIBRATION_POLICY_ID),
                        "calibration_sha256": calibration.calibration_sha256,
                    }
                )
            )
            training_cutoff = shift_month(refit_signal_date.strftime("%Y-%m"), -1)
        elif strategy_id == ELASTIC_NET_STRATEGY_ID:
            native_sha, refit_signal_date = native_identity(("elastic_net",))
            if calibration is None or calibration.fit_signal_date != refit_signal_date:
                raise Stage13ProductionRuntimeError(
                    "elastic-net model and shared calibration refit dates differ"
                )
            refit_id = _sha256_json(
                {
                    "native_state_sha256": native_sha,
                    "shared_calibration_policy_id": SHARED_CALIBRATION_POLICY_ID,
                    "calibration_sha256": calibration.calibration_sha256,
                }
            )
            training_cutoff = shift_month(refit_signal_date.strftime("%Y-%m"), -1)
        elif strategy_id == VQM_STRATEGY_ID:
            if (
                calibration is None
                or calibration.feature_scope is not SHARED_CALIBRATION_SOURCE_SCOPE
            ):
                raise Stage13ProductionRuntimeError(
                    "VQM shared calibration authority differs"
                )
            refit_id = _sha256_json(
                {
                    "fixed_signal_policy": model_config,
                    "shared_calibration_policy_id": SHARED_CALIBRATION_POLICY_ID,
                    "calibration_sha256": calibration.calibration_sha256,
                }
            )
            training_cutoff = calibration.last_holding_month
        elif strategy_id == TRAILING_RESIDUAL_STRATEGY_ID:
            refit_id = _sha256_json(
                {
                    "fixed_signal_policy": model_config,
                    "completed_labels_only": True,
                }
            )
            training_cutoff = trailing_training_cutoff_month
        elif strategy_id == EQUAL_WEIGHT_STRATEGY_ID:
            refit_id = _sha256_json(
                {
                    "fixed_descriptive_policy": model_config,
                    "predictive_model": False,
                }
            )
            training_cutoff = shift_month(holding_month, -1)
        else:  # pragma: no cover - exact strategy set checked above
            raise Stage13ProductionRuntimeError(
                "model-evidence strategy differs from the frozen set"
            )
        if training_cutoff >= holding_month:
            raise Stage13ProductionRuntimeError(
                "model-evidence training cutoff crossed its holding month"
            )
        refit_receipt = _sha256_json(
            {
                "model_config_id": model_config,
                "refit_id": refit_id,
                "training_cutoff_month": training_cutoff,
            }
        )
        strategy_rows: list[dict[str, object]] = []
        for position in np.flatnonzero(admitted):
            permno = int(scored.permno[position])
            forecast_position = forecast_index.get(permno)
            if forecast_position is None:
                raise Stage13ProductionRuntimeError(
                    "model-evidence score escaped the forecast universe"
                )
            economic_score = float(scored.score[position])
            # These fields describe predictive evidence, not the smaller
            # solver candidate set.  Primary/rank have no coherent pre-rank
            # cross-family raw scale; the frozen ensemble score is their raw
            # economic signal.  Elastic net retains its model-scale raw
            # forecast, trailing residual retains its direct return estimate,
            # VQM is already once-ranked, and equal weight has no forecast.
            if strategy_id in {PRIMARY_STRATEGY_ID, RANK_STRATEGY_ID}:
                raw = float(forecast.primary_ensemble_score[forecast_position])
                if economic_score != raw:
                    raise Stage13ProductionRuntimeError(
                        "primary/rank scored evidence differs from its forecast receipt"
                    )
            elif strategy_id == ELASTIC_NET_STRATEGY_ID:
                raw = float(forecast.elastic_net_raw[forecast_position])
                if economic_score != float(
                    forecast.elastic_net_score[forecast_position]
                ):
                    raise Stage13ProductionRuntimeError(
                        "elastic-net scored evidence differs from its forecast receipt"
                    )
            elif strategy_id == TRAILING_RESIDUAL_STRATEGY_ID:
                try:
                    raw = float(direct_expected_return_by_permno[permno])
                except KeyError as exc:
                    raise Stage13ProductionRuntimeError(
                        "admitted trailing-residual score lacks its direct estimate"
                    ) from exc
            elif strategy_id == VQM_STRATEGY_ID:
                raw = economic_score
            elif strategy_id == EQUAL_WEIGHT_STRATEGY_ID:
                # The carried ScoredSnapshot is universe plumbing only.  The
                # descriptive equal-weight book has no predictive signal.
                raw = 0.0
                economic_score = 0.0
            else:  # pragma: no cover - exact strategy set checked above
                raise Stage13ProductionRuntimeError(
                    "model-evidence strategy differs from the frozen set"
                )
            calibrated = raw if slope is None else slope * economic_score
            if not all(math.isfinite(value) for value in (raw, calibrated)):
                raise Stage13ProductionRuntimeError(
                    "model-evidence forecast is nonfinite"
                )
            forecast_receipt = _sha256_json(
                {
                    "holding_month": holding_month,
                    "strategy_id": strategy_id,
                    "permno": permno,
                    "raw_forecast": raw,
                    "calibration_slope": slope,
                    "calibrated_forecast": calibrated,
                    "score": economic_score,
                    "refit_receipt_sha256": refit_receipt,
                }
            )
            strategy_rows.append(
                {
                    "holding_month": holding_month,
                    "segment": segment.value,
                    "strategy_id": strategy_id,
                    "permno": permno,
                    "model_config_id": model_config,
                    "refit_id": refit_id,
                    "training_cutoff_month": training_cutoff,
                    "refit_receipt_sha256": refit_receipt,
                    "raw_forecast": raw,
                    "calibration_slope": slope,
                    "calibrated_forecast": calibrated,
                    "score": economic_score,
                    "forecast_receipt_sha256": forecast_receipt,
                    "monthly_spearman_ic": 0.0,
                    "checkpoint_sha256": checkpoint_sha256,
                }
            )
        ic = _monthly_ic(
            strategy_rows,
            labels=labels,
            outcomes=outcomes,
            partial=segment is Stage13Segment.PARTIAL,
        )
        for row in strategy_rows:
            row["monthly_spearman_ic"] = ic
        records.extend(strategy_rows)
    return records


def _compound(values: Sequence[object]) -> float:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or not np.isfinite(vector).all():
        raise Stage13ProductionRuntimeError("daily factor path is nonfinite")
    return float(np.prod(1.0 + vector, dtype=np.float64) - 1.0)


def _factor_and_benchmark_fragment(
    *,
    segment: Stage13Segment,
    holding_month: str,
    factor_frame: pd.DataFrame | None,
    benchmark_frame: pd.DataFrame,
    accounting_table: pa.Table,
    equal_weight_result: StrategyMonthResult,
    checkpoint_sha256: str,
) -> Mapping[str, object]:
    if factor_frame is not None:
        if len(factor_frame) != 1:
            raise Stage13ProductionRuntimeError(
                "final monthly factor row is absent or duplicated"
            )
        source = factor_frame.iloc[0]
        factor = {
            "risk_free_return": float(source["rf"]),
            **{name: float(source[name]) for name in FACTOR_NAMES},
        }
    else:
        daily = accounting_table.select(
            ["calendar_date", *FACTOR_NAMES[:-1], "rf", "umd"]
        ).to_pandas()
        daily = daily.drop_duplicates().sort_values("calendar_date", kind="mergesort")
        if daily["calendar_date"].duplicated().any() or daily.empty:
            raise Stage13ProductionRuntimeError(
                "partial daily factor path is absent or inconsistent"
            )
        factor = {
            "risk_free_return": _compound(daily["rf"]),
            **{name: _compound(daily[name]) for name in FACTOR_NAMES},
        }
    factor_row = {
        "holding_month": holding_month,
        "segment": segment.value,
        **factor,
        "checkpoint_sha256": checkpoint_sha256,
    }
    if benchmark_frame.empty or benchmark_frame["benchmark_id"].duplicated().any():
        raise Stage13ProductionRuntimeError(
            "final benchmark month is empty or duplicated"
        )
    observed = {
        str(row.benchmark_id): float(row.total_return)
        for row in benchmark_frame.itertuples(index=False)
    }
    passive = set(DESCRIPTIVE_BENCHMARK_IDS) - {EQUAL_WEIGHT_STRATEGY_ID}
    if set(observed) != passive:
        raise Stage13ProductionRuntimeError(
            "registered final passive benchmark set differs"
        )
    observed[EQUAL_WEIGHT_STRATEGY_ID] = equal_weight_result.net_total_return
    benchmark_rows = [
        {
            "holding_month": holding_month,
            "segment": segment.value,
            "benchmark_id": benchmark_id,
            "total_return": observed[benchmark_id],
            "active_turnover_cost_applied": (benchmark_id == EQUAL_WEIGHT_STRATEGY_ID),
        }
        for benchmark_id in DESCRIPTIVE_BENCHMARK_IDS
    ]
    return MappingProxyType(
        {"factor_rows": [factor_row], "benchmark_rows": benchmark_rows}
    )


def _access_fragment_rows(
    *,
    segment: Stage13Segment,
    holding_month: str,
    stage12_binding_sha256: str,
    events: Sequence[object],
    checkpoint_sha256: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for event in events:
        payload = {
            "segment": segment.value,
            "holding_month": holding_month,
            "registration_name": str(event.registration_name),
            "stage12_binding_sha256": stage12_binding_sha256,
            "checkpoint_sha256": checkpoint_sha256,
            "file_sha256": str(event.sha256),
            "row_count": int(event.row_count),
        }
        rows.append({**payload, "access_receipt_sha256": _sha256_json(payload)})
    return rows


def _rebind_fragments_to_checkpoint(
    product: ProductionMonthProduct,
    *,
    checkpoint_sha256: str,
) -> Mapping[str, Mapping[str, object]]:
    fragments = json.loads(result_json_bytes(dict(product.result_fragments)))
    for name in (
        MODEL_EVIDENCE_FILE,
        DAILY_ACCOUNTING_FILE,
        MONTHLY_ACCOUNTING_FILE,
        COST_CONSTRAINT_MATERIALITY_FILE,
    ):
        for row in fragments[name]["records"]:
            row["checkpoint_sha256"] = checkpoint_sha256
    for field in (
        "terminal_event_sources",
        "terminal_event_strategy_observations",
    ):
        for row in fragments[COST_CONSTRAINT_MATERIALITY_FILE][field]:
            row["checkpoint_sha256"] = checkpoint_sha256
    allocation = fragments[ALLOCATION_EVIDENCE_FILE]
    for row in allocation["records"]:
        row["checkpoint_sha256"] = checkpoint_sha256
        problem = _sha256_json(
            {
                "segment": row["segment"],
                "holding_month": row["holding_month"],
                "strategy_id": row["strategy_id"],
                "checkpoint_sha256": checkpoint_sha256,
                "policy_id": "frozen-stage12-allocation-policy-v1",
            }
        )
        row["allocation_problem_sha256"] = problem
        row["constraint_certificate_sha256"] = _sha256_json(
            {
                "allocation_problem_sha256": problem,
                "permno": row["permno"],
                "current_weight": row["current_weight"],
                "target_weight": row["target_weight"],
                "trade_weight": row["trade_weight"],
            }
        )
    candidate_fields = (
        "holding_month",
        "segment",
        "strategy_id",
        "eligible_universe_count",
        "candidate_long_count",
        "candidate_short_count",
        "reduction_only_count",
        "minimum_new_names_each_side",
        "minimum_each_side_applies",
        "tail_extension_used",
        "insufficient_side_fallback_used",
        "new_risk_permitted",
        "candidate_sets_disjoint",
        "checkpoint_sha256",
    )
    for row in allocation["candidate_audits"]:
        row["checkpoint_sha256"] = checkpoint_sha256
        row["candidate_evidence_sha256"] = _sha256_json(
            {field: row[field] for field in candidate_fields}
        )
    for row in fragments[FACTOR_BENCHMARK_FILE]["factor_rows"]:
        row["checkpoint_sha256"] = checkpoint_sha256
    rebound_access: list[dict[str, object]] = []
    for row in fragments[ACCESS_LEDGER_FILE]["records"]:
        payload = {
            "segment": row["segment"],
            "holding_month": row["holding_month"],
            "registration_name": row["registration_name"],
            "stage12_binding_sha256": row["stage12_binding_sha256"],
            "checkpoint_sha256": checkpoint_sha256,
            "file_sha256": row["file_sha256"],
            "row_count": row["row_count"],
        }
        rebound_access.append(
            {**payload, "access_receipt_sha256": _sha256_json(payload)}
        )
    fragments[ACCESS_LEDGER_FILE]["records"] = rebound_access
    return MappingProxyType(
        {
            name: MappingProxyType(dict(fragments[name]))
            for name in (
                MODEL_EVIDENCE_FILE,
                ALLOCATION_EVIDENCE_FILE,
                DAILY_ACCOUNTING_FILE,
                MONTHLY_ACCOUNTING_FILE,
                COST_CONSTRAINT_MATERIALITY_FILE,
                FACTOR_BENCHMARK_FILE,
                ACCESS_LEDGER_FILE,
            )
        }
    )


def _authenticated_stage12_trial_ledger(
    binding: Stage12VerifiedBinding,
) -> Mapping[str, Any]:
    pointer_path = (
        PROJECT_ROOT / "artifacts/phase_06_final_test/LATEST_STAGE_12_PRIMARY.json"
    )
    pointer_payload = _stable_bound_bytes(
        pointer_path,
        trusted_root=PROJECT_ROOT,
        expected_sha256=None,
        maximum_size_bytes=1024 * 1024,
        label="Stage 12 primary pointer for trial ledger",
    )
    pointer = _json_mapping(pointer_payload, label="Stage 12 primary pointer")
    if (
        pointer.get("freeze_id") != binding.build_id
        or pointer.get("manifest_sha256") != binding.manifest_sha256
        or not isinstance(pointer.get("bundle_path"), str)
    ):
        raise Stage13ProductionRuntimeError(
            "Stage 12 trial-ledger primary pointer differs"
        )
    bundle = PROJECT_ROOT / str(pointer["bundle_path"])
    manifest_payload = _stable_bound_bytes(
        bundle / "pre_final_manifest.json",
        trusted_root=PROJECT_ROOT,
        expected_sha256=binding.manifest_sha256,
        maximum_size_bytes=8 * 1024 * 1024,
        label="Stage 12 manifest for trial ledger",
    )
    manifest = _json_mapping(manifest_payload, label="Stage 12 manifest")
    member = manifest.get("members", {}).get("trial_ledger.json")  # type: ignore[union-attr]
    if not isinstance(member, Mapping):
        raise Stage13ProductionRuntimeError(
            "Stage 12 manifest omits the frozen trial ledger"
        )
    ledger_payload = _stable_bound_bytes(
        bundle / "trial_ledger.json",
        trusted_root=PROJECT_ROOT,
        expected_sha256=_require_hash(
            member.get("sha256"), field="stage12_trial_ledger_sha256"
        ),
        expected_size_bytes=int(member.get("size_bytes", -1)),
        maximum_size_bytes=64 * 1024 * 1024,
        label="Stage 12 frozen trial ledger",
    )
    return _json_mapping(ledger_payload, label="Stage 12 frozen trial ledger")


class _AuthenticatedNativeProductionAdapter(_ProductionAdapter):
    """The sole empirical Stage 13 month adapter.

    Final payloads are opened only by the live broker.  Every state required
    for exact continuation is persisted before the enclosing monthly
    checkpoint is committed, so a restart never refits or re-accounts an
    authenticated completed month.
    """

    def __init__(
        self,
        *,
        binding: Stage12VerifiedBinding,
        approval: Stage13ApprovalReceipt,
        broker: FinalAccessBroker,
        history: AuthenticatedPrefinalTrainingHistory,
    ) -> None:
        if not isinstance(binding, Stage12VerifiedBinding):
            raise Stage13ProductionRuntimeError(
                "native adapter Stage 12 binding differs"
            )
        authenticated = require_stage13_unblind_authorization(
            approval, expected_stage12_binding=binding
        )
        if type(broker) is not FinalAccessBroker or not isinstance(
            history, AuthenticatedPrefinalTrainingHistory
        ):
            raise Stage13ProductionRuntimeError(
                "native adapter authenticated authorities differ"
            )
        if history.stage12_binding_sha256 != binding.binding_sha256:
            raise Stage13ProductionRuntimeError(
                "native adapter prefinal history binding differs"
            )
        self.binding = binding
        self.approval = authenticated
        self.broker = broker
        self.history = history
        self.native_store = Stage13NativeStateStore(
            (STAGE13_CHECKPOINT_ROOT / "_native_model_states").absolute()
        )
        self.adapter_store = Stage13AdapterStateStore(
            (STAGE13_CHECKPOINT_ROOT / "_adapter_states").absolute()
        )
        self._strategy_states: dict[FeatureScope, dict[str, _NativeStrategyState]] = {
            scope: {
                strategy_id: _NativeStrategyState(
                    portfolio=PortfolioState.all_cash(REFERENCE_NAV_DOLLARS),
                    action_ledger=CorporateActionLedger(),
                    mark_ledger=MarkContinuityLedger.empty(),
                )
                for strategy_id in ACTIVE_STRATEGY_IDS
            }
            for scope in FeatureScope
        }
        self._risk_memory: dict[FeatureScope, dict[int, _NativeRiskMemory]] = {
            scope: {} for scope in FeatureScope
        }
        self._retired_on: dict[int, dt.date] = {}
        self._session_dates: tuple[dt.date, ...] = ()
        self._model_states: dict[str, Stage13NativeFamilyState] = {}
        self._calibrations: dict[str, FinalCalibrationReceipt] = {}
        self._kernel_state: FinalKernelState | None = None
        self._feature_refs: dict[str, dict[str, str]] = {
            scope.value: {} for scope in FeatureScope
        }
        self._label_refs: dict[str, str] = {}
        self._score_refs: dict[str, dict[str, str]] = {
            scope.value: {} for scope in FeatureScope
        }
        self._prefinal_full_features: pd.DataFrame | None = None
        self._prefinal_labels: pd.DataFrame | None = None
        self._prefinal_model_scores: pd.DataFrame | None = None
        self._prefinal_calibration: dict[
            tuple[FeatureScope, str], tuple[FinalCalibrationObservation, ...]
        ] = {}

    def _request(self, registration_name: str, holding_month: str) -> object:
        try:
            role = _REGISTRATION_ROLE[registration_name]
            columns = FROZEN_EXACT_COLUMNS[registration_name]
        except KeyError as exc:
            raise Stage13ProductionRuntimeError(
                "native adapter registration escaped the frozen plan"
            ) from exc
        return self.broker.request_table(
            registration_name,
            role=role,
            holding_months=(holding_month,),
            columns=columns,
        )

    def _read(self, registration_name: str, holding_month: str) -> tuple[pa.Table, str]:
        request = self._request(registration_name, holding_month)
        table = self.broker.read_guarded_parquet(request)  # type: ignore[arg-type]
        return table, str(request.record.sha256)  # type: ignore[attr-defined]

    def _prefinal_features(self, scope: FeatureScope) -> pd.DataFrame:
        if self._prefinal_full_features is None:
            # Exactly one Stage 11 validation-predictor broker read.  The core
            # scope is a column projection of this same authenticated frame.
            self._prefinal_full_features = self.history.read_prefinal_features(
                FeatureScope.FULL_FEATURE_PRIMARY
            )
        if scope is FeatureScope.FULL_FEATURE_PRIMARY:
            return self._prefinal_full_features.copy()
        columns = tuple(
            dict.fromkeys((*FEATURE_SPINE_COLUMNS, *self.history.feature_names(scope)))
        )
        return self._prefinal_full_features.loc[:, list(columns)].copy()

    def _prefinal_completed_labels(self) -> pd.DataFrame:
        if self._prefinal_labels is None:
            self._prefinal_labels = self.history.read_prefinal_labels()
        return self._prefinal_labels.copy()

    def _prefinal_scores(self) -> pd.DataFrame:
        if self._prefinal_model_scores is None:
            self._prefinal_model_scores = self.history.read_prefinal_model_scores()
        return self._prefinal_model_scores.copy()

    def _frames(self, references: Mapping[str, str]) -> pd.DataFrame:
        if not references:
            return pd.DataFrame()
        frames = [
            self.adapter_store.load_frame(references[month])
            for month in sorted(references)
        ]
        return pd.concat(frames, ignore_index=True)

    def _final_features(self, scope: FeatureScope) -> pd.DataFrame:
        return self._frames(self._feature_refs[scope.value])

    def _final_completed_labels(self) -> pd.DataFrame:
        return self._frames(self._label_refs)

    def _final_scores(self, scope: FeatureScope) -> pd.DataFrame:
        return self._frames(self._score_refs[scope.value])

    def _state_payload(self, *, holding_month: str) -> Mapping[str, object]:
        if self._kernel_state is None:
            raise Stage13ProductionRuntimeError(
                "native adapter cannot persist an absent kernel state"
            )
        self._require_shared_calibration_state(self._calibrations)
        return {
            "schema_version": _NATIVE_ADAPTER_STATE_SCHEMA_VERSION,
            "holding_month": holding_month,
            "stage12_binding_sha256": self.binding.binding_sha256,
            "kernel_state": self._kernel_state.payload(),
            "model_state_sha256": {
                key: self._model_states[key].state_sha256 for key in MODEL_STATE_KEYS
            },
            "calibrations": {
                key: self._calibrations[key].payload() for key in CALIBRATION_STATE_KEYS
            },
            "strategy_states": {
                scope.value: {
                    strategy_id: self._strategy_states[scope][strategy_id].payload()
                    for strategy_id in ACTIVE_STRATEGY_IDS
                }
                for scope in FeatureScope
            },
            "risk_memory": {
                scope.value: {
                    str(permno): memory.payload()
                    for permno, memory in sorted(self._risk_memory[scope].items())
                }
                for scope in FeatureScope
            },
            "retired_identity_dates": _retired_identity_state_payload(self._retired_on),
            "completed_session_dates": [
                value.isoformat() for value in self._session_dates
            ],
            "feature_frame_sha256_by_scope_month": {
                scope: dict(sorted(values.items()))
                for scope, values in self._feature_refs.items()
            },
            "label_frame_sha256_by_month": dict(sorted(self._label_refs.items())),
            "score_frame_sha256_by_scope_month": {
                scope: dict(sorted(values.items()))
                for scope, values in self._score_refs.items()
            },
        }

    @staticmethod
    def _hash_map(value: object, *, field: str) -> dict[str, str]:
        if not isinstance(value, Mapping):
            raise Stage13ProductionRuntimeError(f"persisted {field} differs")
        return {
            str(key): _require_hash(item, field=f"{field}.{key}")
            for key, item in value.items()
        }

    def restore(self, product: ProductionMonthProduct) -> None:
        if not isinstance(product, ProductionMonthProduct):
            raise Stage13ProductionRuntimeError(
                "native adapter restore product differs"
            )
        state = self.adapter_store.load_state(product.adapter_state_sha256)
        expected = {
            "schema_version",
            "holding_month",
            "stage12_binding_sha256",
            "kernel_state",
            "model_state_sha256",
            "calibrations",
            "strategy_states",
            "risk_memory",
            "retired_identity_dates",
            "completed_session_dates",
            "feature_frame_sha256_by_scope_month",
            "label_frame_sha256_by_month",
            "score_frame_sha256_by_scope_month",
        }
        if (
            set(state) != expected
            or state.get("schema_version") != _NATIVE_ADAPTER_STATE_SCHEMA_VERSION
            or state.get("holding_month") != product.holding_month
            or state.get("stage12_binding_sha256") != self.binding.binding_sha256
            or not isinstance(state.get("kernel_state"), Mapping)
        ):
            raise Stage13ProductionRuntimeError(
                "persisted native adapter state envelope differs"
            )
        kernel = FinalKernelState.from_mapping(state["kernel_state"])
        if kernel.state_sha256 != product.kernel_state_sha256:
            raise Stage13ProductionRuntimeError(
                "persisted native kernel state differs from checkpoint product"
            )
        model_hashes = self._hash_map(
            state["model_state_sha256"], field="model_state_sha256"
        )
        if model_hashes != dict(product.model_state_sha256):
            raise Stage13ProductionRuntimeError(
                "persisted native model state differs from checkpoint product"
            )
        self._model_states = {
            key: self.native_store.load(model_hashes[key]) for key in MODEL_STATE_KEYS
        }
        raw_calibrations = state["calibrations"]
        if not isinstance(raw_calibrations, Mapping) or set(raw_calibrations) != set(
            CALIBRATION_STATE_KEYS
        ):
            raise Stage13ProductionRuntimeError(
                "persisted final calibration state set differs"
            )
        self._calibrations = {
            key: _calibration_from_payload(value)
            for key, value in raw_calibrations.items()
            if isinstance(value, Mapping)
        }
        if set(self._calibrations) != set(CALIBRATION_STATE_KEYS):
            raise Stage13ProductionRuntimeError(
                "persisted final calibration payload differs"
            )
        self._require_shared_calibration_state(self._calibrations)
        raw_strategies = state["strategy_states"]
        if not isinstance(raw_strategies, Mapping) or set(raw_strategies) != {
            scope.value for scope in FeatureScope
        }:
            raise Stage13ProductionRuntimeError(
                "persisted strategy-scope state differs"
            )
        restored_strategies: dict[FeatureScope, dict[str, _NativeStrategyState]] = {}
        for scope in FeatureScope:
            values = raw_strategies[scope.value]
            if not isinstance(values, Mapping) or set(values) != set(
                ACTIVE_STRATEGY_IDS
            ):
                raise Stage13ProductionRuntimeError(
                    "persisted strategy state set differs"
                )
            restored_strategies[scope] = {
                strategy_id: _NativeStrategyState.from_payload(values[strategy_id])
                for strategy_id in ACTIVE_STRATEGY_IDS
                if isinstance(values[strategy_id], Mapping)
            }
            if set(restored_strategies[scope]) != set(ACTIVE_STRATEGY_IDS):
                raise Stage13ProductionRuntimeError(
                    "persisted strategy payload differs"
                )
        raw_memory = state["risk_memory"]
        if not isinstance(raw_memory, Mapping) or set(raw_memory) != {
            scope.value for scope in FeatureScope
        }:
            raise Stage13ProductionRuntimeError("persisted risk-memory scope differs")
        restored_memory: dict[FeatureScope, dict[int, _NativeRiskMemory]] = {}
        for scope in FeatureScope:
            values = raw_memory[scope.value]
            if not isinstance(values, Mapping):
                raise Stage13ProductionRuntimeError(
                    "persisted risk-memory payload differs"
                )
            restored_memory[scope] = {
                int(permno): _NativeRiskMemory.from_payload(payload)
                for permno, payload in values.items()
                if isinstance(payload, Mapping)
            }
            if set(restored_memory[scope]) != {int(value) for value in values}:
                raise Stage13ProductionRuntimeError(
                    "persisted risk-memory identity differs"
                )
        try:
            session_dates = tuple(
                dt.date.fromisoformat(str(value))
                for value in state["completed_session_dates"]  # type: ignore[union-attr]
            )
        except (TypeError, ValueError) as exc:
            raise Stage13ProductionRuntimeError(
                "persisted completed-session history differs"
            ) from exc
        if session_dates != tuple(sorted(set(session_dates))):
            raise Stage13ProductionRuntimeError(
                "persisted completed-session history is not ordered and unique"
            )
        retired_on = _retired_identity_state_from_payload(
            state["retired_identity_dates"],
            completed_session_dates=session_dates,
        )
        feature_refs = state["feature_frame_sha256_by_scope_month"]
        score_refs = state["score_frame_sha256_by_scope_month"]
        if not isinstance(feature_refs, Mapping) or not isinstance(score_refs, Mapping):
            raise Stage13ProductionRuntimeError(
                "persisted final frame reference scopes differ"
            )
        self._feature_refs = {
            scope.value: self._hash_map(
                feature_refs[scope.value], field=f"feature_refs.{scope.value}"
            )
            for scope in FeatureScope
        }
        self._score_refs = {
            scope.value: self._hash_map(
                score_refs[scope.value], field=f"score_refs.{scope.value}"
            )
            for scope in FeatureScope
        }
        self._label_refs = self._hash_map(
            state["label_frame_sha256_by_month"], field="label_refs"
        )
        self._strategy_states = restored_strategies
        self._risk_memory = restored_memory
        self._retired_on = retired_on
        self._session_dates = session_dates
        self._kernel_state = kernel
        reported_scope = _reported_scope(product.segment)
        closing = {
            strategy_id: _portfolio_sha(
                self._strategy_states[reported_scope][strategy_id].portfolio
            )
            for strategy_id in ACTIVE_STRATEGY_IDS
        }
        if closing != dict(product.closing_book_sha256_by_strategy):
            raise Stage13ProductionRuntimeError(
                "restored strategy books differ from checkpoint product"
            )
        if product.segment is Stage13Segment.PRIMARY:
            shadow = {
                strategy_id: _portfolio_sha(
                    self._strategy_states[FeatureScope.REDUCED_AUXILIARY_CORE][
                        strategy_id
                    ].portfolio
                )
                for strategy_id in ACTIVE_STRATEGY_IDS
            }
            if shadow != dict(product.shadow_core_closing_book_sha256_by_strategy):
                raise Stage13ProductionRuntimeError(
                    "restored shadow-core books differ from checkpoint product"
                )

    @staticmethod
    def _make_calibration_observations(
        *,
        scope: FeatureScope,
        strategy_id: str,
        score_frame: pd.DataFrame,
        score_column: str,
        labels: pd.DataFrame,
        source: str,
        fit_signal_date: dt.date | None = None,
        fit_holding_month: str | None = None,
    ) -> tuple[FinalCalibrationObservation, ...]:
        score_fields = ["signal_date", "holding_month", "permno", score_column]
        missing = sorted(set(score_fields) - set(score_frame))
        if missing:
            raise Stage13ProductionRuntimeError(
                "calibration score history omits fields: " + ", ".join(missing)
            )
        label_fields = [
            "signal_date",
            "holding_month",
            "entry_date",
            "exit_date",
            "outcome_availability_date",
            "permno",
            "unranked_residual_return",
            "primary_label_status",
        ]
        missing = sorted(set(label_fields) - set(labels))
        if missing:
            raise Stage13ProductionRuntimeError(
                "calibration label history omits fields: " + ", ".join(missing)
            )
        left = score_frame.loc[:, score_fields].copy()
        right = labels.loc[:, label_fields].copy()
        merged = left.merge(
            right,
            on=["signal_date", "holding_month", "permno"],
            how="inner",
            validate="one_to_one",
        )
        merged[score_column] = pd.to_numeric(merged[score_column], errors="coerce")
        merged["unranked_residual_return"] = pd.to_numeric(
            merged["unranked_residual_return"], errors="coerce"
        )
        merged = merged.loc[
            merged["primary_label_status"].astype(str).eq(PRESENT)
            & np.isfinite(merged[score_column].to_numpy(float))
            & np.isfinite(merged["unranked_residual_return"].to_numpy(float))
        ].sort_values(["holding_month", "permno"], kind="mergesort")
        if (fit_signal_date is None) != (fit_holding_month is None):
            raise Stage13ProductionRuntimeError(
                "calibration cutoff inputs must be supplied together"
            )
        if fit_signal_date is not None:
            merged = merged.loc[
                merged["exit_date"].le(fit_signal_date)
                & merged["outcome_availability_date"].le(fit_signal_date)
                & merged["holding_month"].astype(str).lt(str(fit_holding_month))
            ].copy()
        forecast_payload = _frame_sha256(left)
        result: list[FinalCalibrationObservation] = []
        for row in merged.itertuples(index=False):
            result.append(
                FinalCalibrationObservation(
                    feature_scope=scope,
                    strategy_id=strategy_id,
                    forecast_id=f"{source}:{strategy_id}",
                    holding_month=str(row.holding_month),
                    permno=int(row.permno),
                    score=float(getattr(row, score_column)),
                    residual_return=float(row.unranked_residual_return),
                    signal_date=row.signal_date,
                    entry_date=row.entry_date,
                    exit_date=row.exit_date,
                    outcome_available_date=row.outcome_availability_date,
                    forecast_payload_sha256=forecast_payload,
                    source=source,
                )
            )
        return tuple(result)

    def _prefinal_calibration_observations(
        self,
        scope: FeatureScope,
        strategy_id: str,
    ) -> tuple[FinalCalibrationObservation, ...]:
        if scope is not FeatureScope.FULL_FEATURE_PRIMARY:
            raise Stage13ProductionRuntimeError(
                "prefinal calibration observations must use the shared "
                "full-feature authority"
            )
        key = (scope, strategy_id)
        if key in self._prefinal_calibration:
            return self._prefinal_calibration[key]
        labels = self._prefinal_completed_labels()
        authenticated_model_domain = self._prefinal_scores()
        if strategy_id == VQM_STRATEGY_ID:
            scores = _vqm_score_frame(self._prefinal_features(scope))
            scores = scores.merge(
                authenticated_model_domain.loc[
                    :, ["signal_date", "holding_month", "permno"]
                ],
                on=["signal_date", "permno"],
                how="inner",
                validate="one_to_one",
            )
            score_column = "vqm_score"
        else:
            scores = authenticated_model_domain
            score_column = (
                "ensemble_score"
                if strategy_id == PRIMARY_STRATEGY_ID
                else "elastic_net_score"
            )
        observations = self._make_calibration_observations(
            scope=scope,
            strategy_id=strategy_id,
            score_frame=scores,
            score_column=score_column,
            labels=labels,
            source="pre_final_pseudo_oof",
        )
        if not observations:
            raise Stage13ProductionRuntimeError(
                "authenticated prefinal calibration history is empty"
            )
        self._prefinal_calibration[key] = observations
        return observations

    def _final_calibration_observations(
        self,
        scope: FeatureScope,
        strategy_id: str,
        *,
        fit_signal_date: dt.date,
        fit_holding_month: str,
    ) -> tuple[FinalCalibrationObservation, ...]:
        if scope is not FeatureScope.FULL_FEATURE_PRIMARY:
            raise Stage13ProductionRuntimeError(
                "final calibration observations must use the shared "
                "full-feature authority"
            )
        scores = self._final_scores(scope)
        labels = self._final_completed_labels()
        if scores.empty or labels.empty:
            return ()
        score_column = {
            PRIMARY_STRATEGY_ID: "ensemble_score",
            ELASTIC_NET_STRATEGY_ID: "elastic_net_score",
            VQM_STRATEGY_ID: "vqm_score",
        }[strategy_id]
        return self._make_calibration_observations(
            scope=scope,
            strategy_id=strategy_id,
            score_frame=scores,
            score_column=score_column,
            labels=labels,
            source="final_prior_live",
            fit_signal_date=fit_signal_date,
            fit_holding_month=fit_holding_month,
        )

    @staticmethod
    def _require_shared_calibration_state(
        calibrations: Mapping[str, FinalCalibrationReceipt],
    ) -> None:
        if (
            SHARED_CALIBRATION_POLICY_ID
            != "full-feature-prefinal-and-prior-live-shared-by-strategy-v1"
            or SHARED_CALIBRATION_SOURCE_SCOPE is not FeatureScope.FULL_FEATURE_PRIMARY
        ):
            raise Stage13ProductionRuntimeError(
                "frozen shared calibration policy differs"
            )
        if set(calibrations) != set(CALIBRATION_STATE_KEYS):
            raise Stage13ProductionRuntimeError(
                "shared final calibration state set differs"
            )
        fit_dates: set[dt.date] = set()
        for strategy_id in CALIBRATED_STRATEGY_IDS:
            full = calibrations[
                f"{FeatureScope.FULL_FEATURE_PRIMARY.value}:{strategy_id}"
            ]
            core = calibrations[
                f"{FeatureScope.REDUCED_AUXILIARY_CORE.value}:{strategy_id}"
            ]
            if (
                full.feature_scope is not FeatureScope.FULL_FEATURE_PRIMARY
                or core.feature_scope is not FeatureScope.FULL_FEATURE_PRIMARY
                or full.strategy_id != strategy_id
                or core.strategy_id != strategy_id
                or full.calibration_sha256 != core.calibration_sha256
                or full.payload() != core.payload()
            ):
                raise Stage13ProductionRuntimeError(
                    "core calibration is not the exact shared full-feature receipt"
                )
            fit_dates.add(full.fit_signal_date)
        if len(fit_dates) != 1:
            raise Stage13ProductionRuntimeError(
                "shared strategy calibrations do not have one fit date"
            )

    def _fit_shared_calibrations(
        self,
        *,
        holding_month: str,
        signal_date: dt.date,
    ) -> None:
        scope = SHARED_CALIBRATION_SOURCE_SCOPE
        existing = self._calibrations.get(f"{scope.value}:{PRIMARY_STRATEGY_ID}")
        if existing is not None and existing.fit_signal_date == signal_date:
            self._require_shared_calibration_state(self._calibrations)
            return
        for strategy_id in CALIBRATED_STRATEGY_IDS:
            observations = (
                *self._prefinal_calibration_observations(scope, strategy_id),
                *self._final_calibration_observations(
                    scope,
                    strategy_id,
                    fit_signal_date=signal_date,
                    fit_holding_month=holding_month,
                ),
            )
            receipt = fit_final_calibration(
                observations,
                feature_scope=scope,
                strategy_id=strategy_id,
                fit_holding_month=holding_month,
                fit_signal_date=signal_date,
            )
            self._calibrations[
                f"{FeatureScope.FULL_FEATURE_PRIMARY.value}:{strategy_id}"
            ] = receipt
            self._calibrations[
                f"{FeatureScope.REDUCED_AUXILIARY_CORE.value}:{strategy_id}"
            ] = receipt
        self._require_shared_calibration_state(self._calibrations)

    def _prepare_refit_and_score(
        self,
        *,
        scope: FeatureScope,
        holding_month: str,
        signal_date: dt.date,
        decision_access_sha256: str,
    ) -> tuple[FinalForecastReceipt, Stage13RefitIdentity | None]:
        final_features = self._final_features(scope)
        final_labels = self._final_completed_labels()
        if final_labels.empty:
            final_labels = pd.DataFrame(
                columns=[
                    "signal_date",
                    "holding_month",
                    "entry_date",
                    "exit_date",
                    "outcome_availability_date",
                    "permno",
                    "unranked_residual_return",
                    "residual_return_rank",
                    "primary_label_status",
                ]
            )
        refit = is_quarterly_refit(holding_month=holding_month, signal_date=signal_date)
        scratch_parent = STAGE13_CHECKPOINT_ROOT / "_native_work"
        scratch_parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(
                prefix=f"{holding_month}-{scope.value}-",
                dir=scratch_parent,
            )
        ).absolute()
        try:
            refit_identity: Stage13RefitIdentity | None = None
            if refit:
                prepared = prepare_stage13_native_refit(
                    history=self.history,
                    scope=scope,
                    holding_month=holding_month,
                    fit_signal_date=signal_date,
                    final_features=final_features,
                    final_completed_labels=final_labels,
                    final_access_ledger_sha256=decision_access_sha256,
                    scratch_root=temporary / "prepare",
                    authenticated_prefinal_features=self._prefinal_features(scope),
                    authenticated_prefinal_labels=(self._prefinal_completed_labels()),
                )
                refit_identity = prepared.identity
                fitted = fit_stage13_selected_native_models(
                    prepared, work_root=temporary / "fit"
                )
                for family, state in fitted.items():
                    key = f"{scope.value}:{family}"
                    self.native_store.persist(state)
                    self._model_states[key] = state
                self._fit_shared_calibrations(
                    holding_month=holding_month,
                    signal_date=signal_date,
                )
                permno = prepared.prepared.score_permnos
                score_base = prepared.prepared.score_base_frozen
                score_elastic = prepared.prepared.score_elastic_frozen
            else:
                states = {
                    family: self._model_states[f"{scope.value}:{family}"]
                    for family in ("neural_network", "xgboost", "elastic_net")
                    if f"{scope.value}:{family}" in self._model_states
                }
                if set(states) != {"neural_network", "xgboost", "elastic_net"}:
                    raise Stage13ProductionRuntimeError(
                        "non-refit final month lacks an active native model state"
                    )
                active_refit = states[
                    "neural_network"
                ].full_preprocessor.refit_signal_date
                combined_features = pd.concat(
                    (self._prefinal_features(scope), final_features),
                    ignore_index=True,
                ).sort_values(["signal_date", "permno"], kind="mergesort")
                combined_labels = pd.concat(
                    (self._prefinal_completed_labels(), final_labels),
                    ignore_index=True,
                )
                permno, score_base, score_elastic = prepare_authenticated_score(
                    refit_signal_date=active_refit,
                    scoring_signal_date=signal_date,
                    combined_features=combined_features,
                    aggregate_history=self.history.aggregate_history(
                        scope, combined_features
                    ),
                    training_labels=combined_labels,
                    feature_dictionary=self.history.feature_dictionary(scope),
                    category_vocabulary=self.history.category_vocabulary(scope),
                    stage9_inputs=self.history.stage9_inputs,
                    full_base_preprocessor=states["neural_network"].full_preprocessor,
                    full_elastic_preprocessor=states["elastic_net"].full_preprocessor,
                    permit_projected_global_ordinals=(
                        scope is FeatureScope.REDUCED_AUXILIARY_CORE
                    ),
                )
            active_states = {
                family: self._model_states[f"{scope.value}:{family}"]
                for family in ("neural_network", "xgboost", "elastic_net")
            }
            forecast = predict_stage13_selected_native_models(
                states=active_states,
                holding_month=holding_month,
                feature_scope=scope,
                permno=permno,
                score_base_frozen=score_base,
                score_elastic_frozen=score_elastic,
                work_root=temporary / "score",
            )
            return forecast, refit_identity
        finally:
            shutil.rmtree(temporary, ignore_errors=True)

    @staticmethod
    def _event_payload(event: object) -> dict[str, object]:
        return {
            "registration_name": str(event.registration_name),
            "role": str(event.role),
            "holding_months": list(event.holding_months),
            "columns": list(event.columns),
            "sha256": str(event.sha256),
            "size_bytes": int(event.size_bytes),
            "row_count": int(event.row_count),
        }

    def execute(self, request: ProductionMonthRequest) -> ProductionMonthProduct:
        if not isinstance(request, ProductionMonthRequest):
            raise Stage13ProductionRuntimeError("native month request type differs")
        plan = request.frozen.plan
        segment = plan.segment
        holding_month = request.frozen.holding_month
        prefix = _SEGMENT_PREFIX[segment]
        if (self._kernel_state is None) != (
            request.previous_kernel_state_sha256 is None
        ) or (
            self._kernel_state is not None
            and self._kernel_state.state_sha256 != request.previous_kernel_state_sha256
        ):
            raise Stage13ProductionRuntimeError(
                "native adapter in-memory predecessor differs from request"
            )

        ledger_start = len(self.broker.ledger().successful_reads)
        input_hashes: dict[str, str] = {}

        calendar_request = self._request("final_calendar", holding_month)
        calendar_issuance = authenticate_final_calendar_row(
            access_broker=self.broker,
            calendar_request=calendar_request,
        )
        input_hashes["final_calendar"] = str(calendar_request.record.sha256)  # type: ignore[attr-defined]
        calendar = calendar_issuance.calendar_row

        predecision_names = (
            f"{prefix}_predictors",
            f"{prefix}_risk_cost",
            f"{prefix}_cost_inputs",
            f"{prefix}_factor_covariance",
            f"{prefix}_risk_cost_month_summary",
            f"{prefix}_market_inputs",
        )
        predecision_tables: dict[str, pa.Table] = {}
        for name in predecision_names:
            table, digest = self._read(name, holding_month)
            predecision_tables[name] = table
            input_hashes[name] = digest
        risk_table = predecision_tables[f"{prefix}_risk_cost"]
        cost_table = predecision_tables[f"{prefix}_cost_inputs"]
        if not risk_table.equals(cost_table, check_metadata=True):
            raise Stage13ProductionRuntimeError(
                "registered final risk and cost inputs differ"
            )

        predictor_frame = _table_frame(
            predecision_tables[f"{prefix}_predictors"],
            date_columns=("signal_date",),
        )
        predictor_frame = predictor_frame.loc[
            predictor_frame["long_eligible"].astype(bool)
        ].copy()
        if (
            predictor_frame.empty
            or predictor_frame["permno"].duplicated().any()
            or set(predictor_frame["signal_date"]) != {calendar.signal_date}
        ):
            raise Stage13ProductionRuntimeError(
                "current final scoring predictor cross-section differs"
            )
        current_features: dict[FeatureScope, pd.DataFrame] = {}
        for scope in _active_scopes(segment):
            columns = tuple(
                dict.fromkeys(
                    (*FEATURE_SPINE_COLUMNS, *self.history.feature_names(scope))
                )
            )
            missing = sorted(set(columns) - set(predictor_frame))
            if missing:
                raise Stage13ProductionRuntimeError(
                    "current final predictor omits scope fields: " + ", ".join(missing)
                )
            frame = (
                predictor_frame.loc[:, list(columns)]
                .copy()
                .sort_values(["signal_date", "permno"], kind="mergesort")
                .reset_index(drop=True)
            )
            frame["permno"] = pd.to_numeric(frame["permno"], errors="raise").astype(
                np.int64
            )
            current_features[scope] = frame
            self._feature_refs[scope.value][holding_month] = (
                self.adapter_store.persist_frame(frame)
            )

        risk_frame = _table_frame(
            risk_table,
            date_columns=(
                "signal_date",
                "information_cutoff_date",
                "risk_window_first_date",
                "risk_window_last_date",
            ),
        )
        market_frame = _table_frame(
            predecision_tables[f"{prefix}_market_inputs"],
            date_columns=("signal_date", "entry_date", "exit_date", "calendar_date"),
        )
        covariance_frame = _table_frame(
            predecision_tables[f"{prefix}_factor_covariance"],
            date_columns=(
                "signal_date",
                "factor_window_first_date",
                "factor_window_last_date",
            ),
        )
        summary_frame = _table_frame(
            predecision_tables[f"{prefix}_risk_cost_month_summary"],
            date_columns=("signal_date",),
        )
        decision_events = self.broker.ledger().successful_reads[ledger_start:]
        decision_access_sha = _sha256_json(
            {
                "stage": 13,
                "holding_month": holding_month,
                "segment": segment.value,
                "calendar_access_receipt_sha256": (
                    calendar_issuance.access_receipt_sha256
                ),
                "events": [self._event_payload(value) for value in decision_events],
                "accounting_or_outcome_rows_opened": False,
            }
        )
        identity = _monthly_identity(
            signal_date=calendar.signal_date,
            holding_month=holding_month,
            decision_inputs=input_hashes,
            calendar_access_receipt_sha256=(calendar_issuance.access_receipt_sha256),
        )
        inherited_identity: dict[int, _NativeRiskMemory] = {}
        for scope in _active_scopes(segment):
            for strategy_state in self._strategy_states[scope].values():
                for permno, value in zip(
                    strategy_state.portfolio.permno,
                    strategy_state.portfolio.position_value_dollars,
                    strict=True,
                ):
                    if float(value) == 0.0:
                        continue
                    security = int(permno)
                    remembered = self._risk_memory[scope].get(security)
                    if remembered is None:
                        raise Stage13ProductionRuntimeError(
                            f"inherited final holding lacks risk memory: {security}"
                        )
                    prior = inherited_identity.get(security)
                    if prior is not None and prior != remembered:
                        raise Stage13ProductionRuntimeError(
                            "inherited final holding risk identity differs across "
                            f"strategies or scopes: {security}"
                        )
                    inherited_identity[security] = remembered
        audit, market, risk = _monthly_market_and_risk(
            identity=identity,
            risk_frame=risk_frame,
            market_frame=market_frame,
            covariance_frame=covariance_frame,
            summary_frame=summary_frame,
            inherited_identity=inherited_identity,
        )
        market = _apply_retired_identity_execution_gate(
            market,
            retired_on=self._retired_on,
            entry_date=calendar.entry_date,
        )
        current_risk_ids = set(
            pd.to_numeric(
                risk_frame.loc[risk_frame["long_eligible"].astype(bool), "permno"],
                errors="raise",
            ).astype(int)
        )
        for scope, frame in current_features.items():
            scored_ids = set(frame["permno"].astype(int))
            if not scored_ids.issubset(current_risk_ids):
                raise Stage13ProductionRuntimeError(
                    f"current {scope.value} score escaped the current risk audit"
                )
        if not set(inherited_identity).issubset(
            set(int(value) for value in market.permno)
        ):
            raise Stage13ProductionRuntimeError(
                "inherited final holding escaped the canonical market union"
            )
        for scope in _active_scopes(segment):
            _update_native_risk_memory(
                self._risk_memory[scope], audit, calendar.signal_date
            )

        forecasts: dict[FeatureScope, FinalForecastReceipt] = {}
        refit_identities: dict[FeatureScope, Stage13RefitIdentity] = {}
        scored_by_scope: dict[FeatureScope, dict[str, ScoredSnapshot]] = {}
        direct_expected: dict[FeatureScope, Mapping[int, float]] = {}
        trailing_cutoff_by_scope: dict[FeatureScope, str] = {}
        all_completed_labels = pd.concat(
            (
                self._prefinal_completed_labels(),
                self._final_completed_labels(),
            ),
            ignore_index=True,
        )
        for scope in _active_scopes(segment):
            forecast, refit_identity = self._prepare_refit_and_score(
                scope=scope,
                holding_month=holding_month,
                signal_date=calendar.signal_date,
                decision_access_sha256=decision_access_sha,
            )
            forecasts[scope] = forecast
            if refit_identity is not None:
                refit_identities[scope] = refit_identity
            vqm = _vqm_score_frame(current_features[scope])
            if not np.array_equal(vqm["permno"].to_numpy(np.int64), forecast.permno):
                raise Stage13ProductionRuntimeError(
                    "VQM and native scoring universes differ"
                )
            vqm_score = vqm["vqm_score"].to_numpy(float)
            trailing_score, trailing_available, expected, trailing_cutoff = (
                _completed_residual_expected_returns(
                    permno=forecast.permno,
                    labels=all_completed_labels,
                    fit_signal_date=calendar.signal_date,
                    fit_holding_month=holding_month,
                )
            )
            direct_expected[scope] = expected
            trailing_cutoff_by_scope[scope] = trailing_cutoff
            current = current_features[scope]
            primary_calibration = self._calibrations[
                f"{scope.value}:{PRIMARY_STRATEGY_ID}"
            ]
            enet_calibration = self._calibrations[
                f"{scope.value}:{ELASTIC_NET_STRATEGY_ID}"
            ]
            vqm_calibration = self._calibrations[f"{scope.value}:{VQM_STRATEGY_ID}"]
            primary_scored = _scored_snapshot(
                identity=identity,
                permno=forecast.permno,
                score=forecast.primary_ensemble_score,
                predictor_frame=current,
                kappa=primary_calibration.kappa,
            )
            scored_by_scope[scope] = {
                PRIMARY_STRATEGY_ID: primary_scored,
                ELASTIC_NET_STRATEGY_ID: _scored_snapshot(
                    identity=identity,
                    permno=forecast.permno,
                    score=forecast.elastic_net_score,
                    predictor_frame=current,
                    kappa=enet_calibration.kappa,
                ),
                RANK_STRATEGY_ID: _scored_snapshot(
                    identity=identity,
                    permno=forecast.permno,
                    score=forecast.primary_ensemble_score,
                    predictor_frame=current,
                    kappa=1.0,
                ),
                TRAILING_RESIDUAL_STRATEGY_ID: _scored_snapshot(
                    identity=identity,
                    permno=forecast.permno,
                    score=trailing_score,
                    predictor_frame=current,
                    kappa=1.0,
                    admitted=trailing_available,
                ),
                VQM_STRATEGY_ID: _scored_snapshot(
                    identity=identity,
                    permno=forecast.permno,
                    score=vqm_score,
                    predictor_frame=current,
                    kappa=vqm_calibration.kappa,
                ),
                EQUAL_WEIGHT_STRATEGY_ID: _scored_snapshot(
                    identity=identity,
                    permno=forecast.permno,
                    score=forecast.primary_ensemble_score,
                    predictor_frame=current,
                    kappa=1.0,
                ),
            }
            score_frame = pd.DataFrame(
                {
                    "signal_date": [calendar.signal_date] * len(forecast.permno),
                    "holding_month": [holding_month] * len(forecast.permno),
                    "permno": forecast.permno,
                    "ensemble_score": forecast.primary_ensemble_score,
                    "elastic_net_score": forecast.elastic_net_score,
                    "vqm_score": vqm_score,
                }
            )
            self._score_refs[scope.value][holding_month] = (
                self.adapter_store.persist_frame(score_frame)
            )

        opening_by_scope: dict[FeatureScope, dict[str, str]] = {
            scope: {
                strategy_id: _portfolio_sha(
                    self._strategy_states[scope][strategy_id].portfolio
                )
                for strategy_id in ACTIVE_STRATEGY_IDS
            }
            for scope in _active_scopes(segment)
        }
        decisions: dict[FeatureScope, dict[str, object]] = {
            scope: {} for scope in _active_scopes(segment)
        }
        # All reported and shadow targets are committed before the accounting
        # registration can be requested or opened.
        for scope in _active_scopes(segment):
            for strategy_id in ACTIVE_STRATEGY_IDS:
                strategy_state = self._strategy_states[scope][strategy_id]
                inherited = _native_inherited_holdings(
                    state=strategy_state.portfolio,
                    identity=identity,
                    audit=audit,
                    memory=self._risk_memory[scope],
                    # Only authenticated sessions from already completed
                    # months are present here. Current sessions are appended
                    # after every decision has been accounted.
                    session_dates=self._session_dates,
                )
                decision_input = StrategyMonthDecisionInput(
                    strategy_id=strategy_id,
                    feature_scope=scope,
                    holding_month=holding_month,
                    signal_date=calendar.signal_date,
                    entry_date=calendar.entry_date,
                    exit_date=calendar.exit_date,
                    scored=scored_by_scope[scope][strategy_id],
                    market=market,
                    inherited=inherited,
                    risk=risk,
                    opening_state=strategy_state.portfolio,
                    decision_access_receipt_sha256=decision_access_sha,
                    expected_return_by_permno=(
                        direct_expected[scope]
                        if strategy_id == TRAILING_RESIDUAL_STRATEGY_ID
                        else None
                    ),
                )
                decisions[scope][strategy_id] = (
                    prepare_certified_strategy_month_decision(decision_input)
                )

        accounting_name = f"{prefix}_market_accounting"
        accounting_request = self._request(accounting_name, holding_month)
        accounting_issuance = authenticate_final_holding_period_session_authority(
            access_broker=self.broker,
            calendar_issuance=calendar_issuance,
            accounting_request=accounting_request,
        )
        input_hashes[accounting_name] = str(accounting_request.record.sha256)  # type: ignore[attr-defined]
        accounting_projection = project_final_daily_accounting_inputs(
            accounting_issuance.accounting_table,
            holding_month=holding_month,
        )
        results_by_scope: dict[FeatureScope, dict[str, StrategyMonthResult]] = {
            scope: {} for scope in _active_scopes(segment)
        }
        for scope in _active_scopes(segment):
            next_states: dict[str, _NativeStrategyState] = {}
            for strategy_id in ACTIVE_STRATEGY_IDS:
                prior = self._strategy_states[scope][strategy_id]
                result = account_certified_strategy_month_decision(
                    decisions[scope][strategy_id],  # type: ignore[arg-type]
                    daily_accounting=accounting_projection,
                    accounting_access_receipt_sha256=(
                        accounting_issuance.access_receipt_sha256
                    ),
                    session_authority=accounting_issuance.authority,
                    initial_action_ledger=prior.action_ledger,
                    initial_mark_continuity_ledger=prior.mark_ledger,
                )
                if (
                    result.closing_action_ledger is None
                    or result.closing_mark_continuity_ledger is None
                ):
                    raise Stage13ProductionRuntimeError(
                        "dated final accounting omitted its restorable ledgers"
                    )
                results_by_scope[scope][strategy_id] = result
                next_states[strategy_id] = _NativeStrategyState(
                    portfolio=result.closing_portfolio.to_state(),
                    action_ledger=result.closing_action_ledger,
                    mark_ledger=result.closing_mark_continuity_ledger,
                )
            self._strategy_states[scope] = next_states

        self._retired_on = _advance_retired_identity_state(
            self._retired_on,
            actions=accounting_projection.corporate_actions,
            entry_date=calendar.entry_date,
            exit_date=calendar.exit_date,
        )

        factor_frame: pd.DataFrame | None = None
        labels_frame: pd.DataFrame | None = None
        if segment is not Stage13Segment.PARTIAL:
            factors_name = f"{prefix}_factors"
            factors_table, digest = self._read(factors_name, holding_month)
            input_hashes[factors_name] = digest
            factor_frame = _table_frame(factors_table)
        benchmark_name = f"{prefix}_benchmarks"
        benchmark_table, digest = self._read(benchmark_name, holding_month)
        input_hashes[benchmark_name] = digest
        benchmark_frame = _table_frame(benchmark_table)
        if segment is Stage13Segment.PARTIAL:
            outcome_name = "partial_descriptive_outcome"
            outcome_table, digest = self._read(outcome_name, holding_month)
            input_hashes[outcome_name] = digest
            outcome_frame = _table_frame(
                outcome_table,
                date_columns=("entry_date", "administrative_close_date"),
            )
        else:
            labels_name = f"{prefix}_completed_labels"
            labels_table, digest = self._read(labels_name, holding_month)
            input_hashes[labels_name] = digest
            labels_frame = _normalized_final_label_history(
                _table_frame(
                    labels_table,
                    date_columns=(
                        "signal_date",
                        "entry_date",
                        "exit_date",
                        "outcome_availability_date",
                    ),
                )
            )
            self._label_refs[holding_month] = self.adapter_store.persist_frame(
                labels_frame
            )
            outcome_name = f"{prefix}_outcomes"
            outcome_table, digest = self._read(outcome_name, holding_month)
            input_hashes[outcome_name] = digest
            outcome_frame = _table_frame(
                outcome_table, date_columns=("entry_date", "exit_date")
            )

        month_events = self.broker.ledger().successful_reads[ledger_start:]
        if (
            len(month_events) != len(plan.registration_names)
            or tuple(value.registration_name for value in month_events)
            != plan.registration_names
            or set(input_hashes) != set(plan.registration_names)
        ):
            raise Stage13ProductionRuntimeError(
                "native month did not open each frozen registration exactly once"
            )
        self._session_dates = tuple(
            sorted(
                set((*self._session_dates, *accounting_issuance.authority.return_dates))
            )
        )
        full_access_sha = _sha256_json(
            {
                "stage12_binding_sha256": self.binding.binding_sha256,
                "approval_payload_sha256": self.approval.payload_sha256,
                "holding_month": holding_month,
                "segment": segment.value,
                "events": [self._event_payload(value) for value in month_events],
            }
        )

        reported_scope = _reported_scope(segment)
        reported_results = tuple(
            results_by_scope[reported_scope][strategy_id]
            for strategy_id in ACTIVE_STRATEGY_IDS
        )
        shadow_results = (
            tuple(
                results_by_scope[FeatureScope.REDUCED_AUXILIARY_CORE][strategy_id]
                for strategy_id in ACTIVE_STRATEGY_IDS
            )
            if segment is Stage13Segment.PRIMARY
            else ()
        )
        result_hashes = {
            value.strategy_id: value.result_sha256 for value in reported_results
        }
        shadow_hashes = {
            value.strategy_id: value.result_sha256 for value in shadow_results
        }
        closing_reported = {
            strategy_id: _portfolio_sha(
                self._strategy_states[reported_scope][strategy_id].portfolio
            )
            for strategy_id in ACTIVE_STRATEGY_IDS
        }
        shadow_opening = (
            opening_by_scope[FeatureScope.REDUCED_AUXILIARY_CORE]
            if segment is Stage13Segment.PRIMARY
            else {}
        )
        shadow_closing = (
            {
                strategy_id: _portfolio_sha(
                    self._strategy_states[FeatureScope.REDUCED_AUXILIARY_CORE][
                        strategy_id
                    ].portfolio
                )
                for strategy_id in ACTIVE_STRATEGY_IDS
            }
            if segment is Stage13Segment.PRIMARY
            else {}
        )
        model_hashes = {
            key: self._model_states[key].state_sha256 for key in MODEL_STATE_KEYS
        }
        calibration_hashes = {
            key: self._calibrations[key].calibration_sha256
            for key in CALIBRATION_STATE_KEYS
        }
        previous_kernel = self._kernel_state
        if previous_kernel is None:
            previous_semantic_sha = _GENESIS_SHA256
            segment_genesis_sha = _GENESIS_SHA256
            prior_heads = None
            prior_shadow_head = None
        elif (
            previous_kernel.segment
            is final_holding_month_descriptor(holding_month).segment
        ):
            previous_semantic_sha = previous_kernel.state_sha256
            segment_genesis_sha = previous_kernel.segment_genesis_sha256
            prior_heads = previous_kernel.pnl_chain_head_sha256_by_strategy
            prior_shadow_head = previous_kernel.shadow_core_head_sha256
        elif segment is Stage13Segment.CORE:
            previous_semantic_sha = previous_kernel.state_sha256
            if previous_kernel.shadow_core_head_sha256 is None:
                raise Stage13ProductionRuntimeError(
                    "core extension lacks an August shadow-core genesis"
                )
            segment_genesis_sha = previous_kernel.shadow_core_head_sha256
            prior_heads = None
            prior_shadow_head = None
        else:
            previous_semantic_sha = previous_kernel.state_sha256
            segment_genesis_sha = previous_kernel.state_sha256
            prior_heads = None
            prior_shadow_head = None
        kernel_state = FinalKernelState.create(
            holding_month=holding_month,
            previous_state_sha256=previous_semantic_sha,
            segment_genesis_sha256=segment_genesis_sha,
            opening_book_sha256_by_strategy=opening_by_scope[reported_scope],
            closing_book_sha256_by_strategy=closing_reported,
            prior_pnl_chain_head_sha256_by_strategy=prior_heads,
            result_sha256_by_strategy=result_hashes,
            model_state_sha256=model_hashes,
            calibration_state_sha256=calibration_hashes,
            shadow_core_opening_book_sha256_by_strategy=(
                shadow_opening if segment is Stage13Segment.PRIMARY else None
            ),
            shadow_core_closing_book_sha256_by_strategy=(
                shadow_closing if segment is Stage13Segment.PRIMARY else None
            ),
            prior_shadow_core_head_sha256=prior_shadow_head,
            shadow_core_result_sha256_by_strategy=(
                shadow_hashes if segment is Stage13Segment.PRIMARY else None
            ),
        )
        if previous_kernel is not None:
            validate_kernel_state_successor(previous_kernel, kernel_state)
        self._kernel_state = kernel_state

        pending_checkpoint = _sha256_json(
            {
                "pending_real_checkpoint": True,
                "segment": segment.value,
                "holding_month": holding_month,
                "kernel_state_sha256": kernel_state.state_sha256,
            }
        )
        kernel_rows = build_stage13_kernel_document_rows(
            reported_results,
            checkpoint_sha256_by_segment_month={
                (segment.value, holding_month): pending_checkpoint
            },
            synthetic=False,
        ).to_documents()
        model_rows = _model_fragment_rows(
            segment=segment,
            holding_month=holding_month,
            scored_by_strategy=scored_by_scope[reported_scope],
            forecast=forecasts[reported_scope],
            direct_expected_return_by_permno=direct_expected[reported_scope],
            trailing_training_cutoff_month=(trailing_cutoff_by_scope[reported_scope]),
            calibrations=self._calibrations,
            model_states=self._model_states,
            labels=labels_frame,
            outcomes=outcome_frame,
            checkpoint_sha256=pending_checkpoint,
        )
        factor_benchmark = _factor_and_benchmark_fragment(
            segment=segment,
            holding_month=holding_month,
            factor_frame=factor_frame,
            benchmark_frame=benchmark_frame,
            accounting_table=accounting_issuance.accounting_table,
            equal_weight_result=results_by_scope[reported_scope][
                EQUAL_WEIGHT_STRATEGY_ID
            ],
            checkpoint_sha256=pending_checkpoint,
        )
        fragments: dict[str, Mapping[str, object]] = {
            MODEL_EVIDENCE_FILE: {"records": model_rows},
            ALLOCATION_EVIDENCE_FILE: {
                "records": kernel_rows[ALLOCATION_EVIDENCE_FILE]["records"],
                "candidate_audits": kernel_rows[ALLOCATION_EVIDENCE_FILE][
                    "candidate_audits"
                ],
            },
            DAILY_ACCOUNTING_FILE: {
                "records": kernel_rows[DAILY_ACCOUNTING_FILE]["records"]
            },
            MONTHLY_ACCOUNTING_FILE: {
                "records": kernel_rows[MONTHLY_ACCOUNTING_FILE]["records"]
            },
            COST_CONSTRAINT_MATERIALITY_FILE: {
                "records": kernel_rows[COST_CONSTRAINT_MATERIALITY_FILE]["records"],
                "terminal_event_sources": kernel_rows[COST_CONSTRAINT_MATERIALITY_FILE][
                    "terminal_event_sources"
                ],
                "terminal_event_strategy_observations": kernel_rows[
                    COST_CONSTRAINT_MATERIALITY_FILE
                ]["terminal_event_strategy_observations"],
            },
            FACTOR_BENCHMARK_FILE: factor_benchmark,
            ACCESS_LEDGER_FILE: {
                "records": _access_fragment_rows(
                    segment=segment,
                    holding_month=holding_month,
                    stage12_binding_sha256=self.binding.binding_sha256,
                    events=month_events,
                    checkpoint_sha256=pending_checkpoint,
                )
            },
        }
        adapter_state_sha = self.adapter_store.persist_state(
            self._state_payload(holding_month=holding_month)
        )
        refit_evidence = (
            _sha256_json(
                {
                    scope.value: refit_identities[scope].identity_sha256
                    for scope in sorted(refit_identities, key=lambda value: value.value)
                }
            )
            if refit_identities
            else None
        )
        forecast_evidence = _sha256_json(
            {
                scope.value: forecasts[scope].receipt_sha256
                for scope in sorted(forecasts, key=lambda value: value.value)
            }
        )
        return ProductionMonthProduct.create(
            segment=segment,
            holding_month=holding_month,
            input_sha256_by_registration=input_hashes,
            access_ledger_sha256=full_access_sha,
            registration_identity_sha256=request.registration_identity_sha256,
            approval_payload_sha256=request.approval_payload_sha256,
            refit_evidence_sha256=refit_evidence,
            forecast_receipt_sha256=forecast_evidence,
            economic_kernel_contract_sha256=(
                FROZEN_FINAL_ECONOMIC_KERNEL_CONTRACT.contract_sha256
            ),
            kernel_previous_state_sha256=request.previous_kernel_state_sha256,
            kernel_state_sha256=kernel_state.state_sha256,
            adapter_state_sha256=adapter_state_sha,
            opening_book_sha256_by_strategy=opening_by_scope[reported_scope],
            closing_book_sha256_by_strategy=closing_reported,
            shadow_core_closing_book_sha256_by_strategy=shadow_closing,
            model_state_sha256=model_hashes,
            result_sha256_by_strategy=result_hashes,
            result_fragments=fragments,
            runtime_id=PRODUCTION_RUNTIME_ID,
        )

    def finalize(
        self,
        run: Stage13RunResult,
        products: tuple[ProductionMonthProduct, ...],
    ) -> Stage13ResultSet:
        if (
            not isinstance(run, Stage13RunResult)
            or run.synthetic
            or not products
            or any(not isinstance(value, ProductionMonthProduct) for value in products)
        ):
            raise Stage13ProductionRuntimeError(
                "native empirical finalization inputs differ"
            )
        checkpoint = {
            (result.plan.segment.value, record.holding_month): (
                record.checkpoint_sha256
            )
            for result in run.segments
            for record in result.chain_records
        }
        fragments: list[EmpiricalMonthResultFragment] = []
        for product in products:
            key = (product.segment.value, product.holding_month)
            if key not in checkpoint:
                raise Stage13ProductionRuntimeError(
                    "native product lacks its committed checkpoint"
                )
            committed = checkpoint[key]
            fragments.append(
                EmpiricalMonthResultFragment.create(
                    segment=product.segment.value,
                    holding_month=product.holding_month,
                    checkpoint_sha256=committed,
                    documents=_rebind_fragments_to_checkpoint(
                        product, checkpoint_sha256=committed
                    ),
                )
            )
        auxiliary = EmpiricalAuxiliaryEvidence.create(
            stage12_trial_ledger=_authenticated_stage12_trial_ledger(self.binding)
        )
        return assemble_empirical_stage13_results(run, fragments, auxiliary)


def _build_authenticated_native_adapter(
    *,
    binding: Stage12VerifiedBinding,
    approval: Stage13ApprovalReceipt,
    broker: FinalAccessBroker,
    history: AuthenticatedPrefinalTrainingHistory,
) -> _ProductionAdapter:
    """Construct the sealed empirical adapter after every authority is live."""

    return _AuthenticatedNativeProductionAdapter(
        binding=binding,
        approval=approval,
        broker=broker,
        history=history,
    )


def run_authenticated_stage13_execution(
    *,
    binding: Stage12VerifiedBinding,
    approval: Stage13ApprovalReceipt,
    broker: FinalAccessBroker,
) -> CompletedStage13Execution:
    """Run or resume the sealed native path after every fixed authority passes."""

    if not isinstance(binding, Stage12VerifiedBinding):
        raise Stage13ProductionRuntimeError("verified Stage 12 binding type differs")
    authenticated = require_stage13_unblind_authorization(
        approval, expected_stage12_binding=binding
    )
    if not isinstance(broker, FinalAccessBroker):
        raise Stage13ProductionRuntimeError("verified final access broker type differs")
    registration_identity = broker.registration_identity
    _registration_identity_sha256(broker)
    history = authenticate_prefinal_training_history(
        binding=binding,
        approval=authenticated,
    )
    adapter = _build_authenticated_native_adapter(
        binding=binding,
        approval=authenticated,
        broker=broker,
        history=history,
    )
    return _run_authenticated_production(
        checkpoint_root=STAGE13_CHECKPOINT_ROOT,
        binding=binding,
        approval=authenticated,
        registration_identity=registration_identity,
        adapter=adapter,
    )


__all__ = [
    "AuthenticatedPrefinalTrainingHistory",
    "CompletedStage13Execution",
    "KERNEL_DETAIL_STATUS",
    "NATIVE_MONTH_ADAPTER_STATUS",
    "PREFINAL_REFIT_INPUT_STATUS",
    "PRODUCTION_RUNTIME_ID",
    "ProductionMonthProduct",
    "ProductionMonthRequest",
    "PreparedStage13NativeRefit",
    "Stage13NativeFamilyState",
    "Stage13NativeStateStore",
    "Stage13AdapterStateStore",
    "Stage13RefitIdentity",
    "Stage13ProductionRuntimeError",
    "authenticate_prefinal_training_history",
    "fit_stage13_selected_native_models",
    "predict_stage13_selected_native_models",
    "prepare_stage13_native_refit",
    "run_authenticated_stage13_execution",
]
