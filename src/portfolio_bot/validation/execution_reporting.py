"""Atomic reader-facing publication for a verified Stage 11 execution.

This module never deserializes or summarizes the validation outcome member; it
only hashes that member as part of authenticating the existing 16-member
execution bundle.  Publication also requires the existing 3-member independent
verification bundle with ``difference_count == 0``.
The immutable report bundle is committed first and the repository-level latest
verified pointer is replaced last.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from portfolio_bot.data.secure_publication import (
    SecurePublicationError,
    SecureStagingDirectory,
    replace_pointer_last,
)
from portfolio_bot.validation.calendar import VALIDATION_HOLDING_MONTHS
from portfolio_bot.validation.execution_orchestrator import (
    DECISION_FOCUSED_CHALLENGER_ID,
    ELASTIC_NET_CHALLENGER_ID,
    PRIMARY_STRATEGY_ID,
    RANK_CHALLENGER_ID,
)
from portfolio_bot.validation.execution_publication import (
    MANIFEST_NAME,
    MONTHLY_METRICS_NAME,
    PRIMARY_MEMBERS,
    PRIMARY_STATUS,
    SELECTION_NAME,
    SUCCESS_NAME,
    VERIFICATION_MEMBERS,
    VERIFIED_STATUS,
    Stage11ExecutionPublication,
    Stage11ExecutionVerification,
)


REPORT_PUBLICATION_ID = "stage11-verified-execution-report-v1"
REPORT_STATUS = "STAGE11_VALIDATION_EXECUTION_REPORT_VERIFIED"
REPORT_SUMMARY_NAME = "summary.json"
REPORT_MARKDOWN_NAME = "report.md"
REPORT_SUCCESS_NAME = "_SUCCESS.json"
REPORT_MEMBERS = (REPORT_SUMMARY_NAME, REPORT_MARKDOWN_NAME, REPORT_SUCCESS_NAME)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts/phase_05_validation"
DEFAULT_REPORT_DIRECTORY = "stage_11_execution_reports"
DEFAULT_LATEST_VERIFIED_POINTER_NAME = "LATEST_STAGE_11_EXECUTION_VERIFIED.json"

_MAX_STAGE11_DATE = pd.Timestamp("2019-12-02")
_HEX64 = frozenset("0123456789abcdef")
_PRIMARY_MEMBER_COUNT = 16
_VERIFICATION_MEMBER_COUNT = 3


class Stage11ExecutionReportingError(RuntimeError):
    """Raised when verified results cannot be authenticated or published."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Stage11ExecutionReportingError(message)


def _canonical_json(value: Any) -> bytes:
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


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value).issubset(_HEX64)
    )


def _stable_file(
    path: Path, *, keep_payload: bool = False
) -> tuple[str, int, bytes | None, tuple[int, ...]]:
    try:
        before = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise Stage11ExecutionReportingError(
            f"cannot open execution authority member: {path.name}"
        ) from exc
    _require(
        stat.S_ISREG(before.st_mode)
        and not stat.S_ISLNK(before.st_mode)
        and before.st_nlink == 1,
        f"unsafe execution authority member: {path.name}",
    )
    digest = hashlib.sha256()
    chunks: list[bytes] | None = [] if keep_payload else None
    try:
        with path.open("rb") as stream:
            while block := stream.read(8 << 20):
                digest.update(block)
                if chunks is not None:
                    chunks.append(block)
        after = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise Stage11ExecutionReportingError(
            f"cannot read execution authority member: {path.name}"
        ) from exc
    fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_nlink",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    _require(
        all(getattr(before, name) == getattr(after, name) for name in fields),
        f"execution authority member changed while read: {path.name}",
    )
    identity = tuple(int(getattr(after, name)) for name in fields)
    return (
        digest.hexdigest(),
        int(after.st_size),
        None if chunks is None else b"".join(chunks),
        identity,
    )


def _json_object(payload: bytes, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Stage11ExecutionReportingError(f"invalid {label} JSON") from exc
    _require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


@dataclass(frozen=True, slots=True)
class _AuthoritySnapshot:
    primary_bundle: Path
    verification_bundle: Path
    identities: Mapping[Path, tuple[int, ...]]
    primary_manifest_sha256: str
    verification_report_sha256: str

    def gate(self) -> None:
        expected_sets = (
            (self.primary_bundle, frozenset(PRIMARY_MEMBERS)),
            (self.verification_bundle, frozenset(VERIFICATION_MEMBERS)),
        )
        for directory, expected in expected_sets:
            try:
                actual = frozenset(path.name for path in directory.iterdir())
            except OSError as exc:
                raise Stage11ExecutionReportingError(
                    "execution authority bundle became unavailable"
                ) from exc
            _require(actual == expected, "execution authority closed set changed")
        fields = (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_nlink",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        for path, expected in self.identities.items():
            try:
                observed = path.stat(follow_symlinks=False)
            except OSError as exc:
                raise Stage11ExecutionReportingError(
                    f"execution authority member disappeared: {path.name}"
                ) from exc
            actual = tuple(int(getattr(observed, name)) for name in fields)
            _require(
                actual == expected
                and stat.S_ISREG(observed.st_mode)
                and not stat.S_ISLNK(observed.st_mode)
                and observed.st_nlink == 1,
                f"execution authority member changed: {path.name}",
            )
        manifest_digest, _, _, _ = _stable_file(
            self.primary_bundle / MANIFEST_NAME
        )
        report_digest, _, _, _ = _stable_file(
            self.verification_bundle / "verification_report.json"
        )
        _require(
            manifest_digest == self.primary_manifest_sha256
            and report_digest == self.verification_report_sha256,
            "execution authority identity changed",
        )


@dataclass(frozen=True, slots=True)
class _AuthenticatedExecutionResults:
    manifest: Mapping[str, Any]
    selection: Mapping[str, Any]
    monthly_metrics: pd.DataFrame
    verification_report: Mapping[str, Any]
    source_member_sha256: Mapping[str, str]
    snapshot: _AuthoritySnapshot


@dataclass(frozen=True, slots=True)
class Stage11VerifiedExecutionReportPublication:
    report_id: str
    bundle_path: Path
    summary_sha256: str
    markdown_sha256: str
    receipt_sha256: str
    pointer_path: Path
    pointer_sha256: str
    reused: bool


def _authenticate_verified_results(
    publication: Stage11ExecutionPublication,
    verification: Stage11ExecutionVerification,
    *,
    expected_synthetic_fixture: bool,
) -> _AuthenticatedExecutionResults:
    _require(
        type(publication) is Stage11ExecutionPublication,
        "reporting requires the exact execution publication receipt",
    )
    _require(
        type(verification) is Stage11ExecutionVerification,
        "reporting requires the exact independent-verification receipt",
    )
    _require(
        verification.difference_count == 0,
        "a zero-difference independent verification is required",
    )
    primary = publication.bundle_path
    verified = verification.bundle_path
    _require(
        primary.is_absolute()
        and verified.is_absolute()
        and primary.is_dir()
        and verified.is_dir()
        and not primary.is_symlink()
        and not verified.is_symlink(),
        "execution authority bundle path is unsafe",
    )
    _require(
        frozenset(path.name for path in primary.iterdir())
        == frozenset(PRIMARY_MEMBERS)
        and len(PRIMARY_MEMBERS) == _PRIMARY_MEMBER_COUNT,
        "primary execution bundle is not the closed 16-member authority",
    )
    _require(
        frozenset(path.name for path in verified.iterdir())
        == frozenset(VERIFICATION_MEMBERS)
        and len(VERIFICATION_MEMBERS) == _VERIFICATION_MEMBER_COUNT,
        "execution verification bundle is not the closed 3-member authority",
    )

    manifest_digest, _, manifest_payload, _ = _stable_file(
        primary / MANIFEST_NAME, keep_payload=True
    )
    assert manifest_payload is not None
    _require(
        manifest_digest == publication.manifest_sha256,
        "primary execution manifest differs from its receipt",
    )
    manifest = _json_object(manifest_payload, label="execution manifest")
    _require(
        manifest.get("build_id") == publication.build_id
        and manifest.get("status") == PRIMARY_STATUS
        and manifest.get("synthetic_fixture") is expected_synthetic_fixture,
        "primary execution manifest identity differs",
    )
    scope = manifest.get("scope")
    _require(
        isinstance(scope, Mapping)
        and scope.get("first_holding_month") == VALIDATION_HOLDING_MONTHS[0]
        and scope.get("last_holding_month") == VALIDATION_HOLDING_MONTHS[-1]
        and scope.get("holding_months") == len(VALIDATION_HOLDING_MONTHS)
        and scope.get("december_2019_holding_month_included") is False
        and scope.get("final_2020_plus_included") is False,
        "execution manifest crossed the Stage 11 validation boundary",
    )
    members = manifest.get("members")
    expected_data_members = frozenset(PRIMARY_MEMBERS) - {
        MANIFEST_NAME,
        SUCCESS_NAME,
    }
    _require(
        isinstance(members, Mapping)
        and frozenset(members) == expected_data_members,
        "execution manifest member authority differs",
    )

    identities: dict[Path, tuple[int, ...]] = {}
    source_hashes: dict[str, str] = {}
    retained: dict[str, bytes] = {}
    for name in PRIMARY_MEMBERS:
        keep = name in {SELECTION_NAME, MONTHLY_METRICS_NAME, SUCCESS_NAME, MANIFEST_NAME}
        digest, size, payload, identity = _stable_file(
            primary / name, keep_payload=keep
        )
        identities[primary / name] = identity
        source_hashes[f"primary/{name}"] = digest
        if payload is not None:
            retained[name] = payload
        if name in expected_data_members:
            record = members.get(name)
            _require(
                isinstance(record, Mapping)
                and record.get("sha256") == digest
                and record.get("size_bytes") == size,
                f"execution member differs from manifest: {name}",
            )
    _require(
        source_hashes[f"primary/{MANIFEST_NAME}"] == publication.manifest_sha256,
        "execution manifest changed during authentication",
    )
    primary_success = _json_object(
        retained[SUCCESS_NAME], label="execution success receipt"
    )
    _require(
        primary_success.get("status") == PRIMARY_STATUS
        and primary_success.get("build_id") == publication.build_id
        and primary_success.get("manifest_sha256") == publication.manifest_sha256
        and primary_success.get("closed_members") == list(PRIMARY_MEMBERS),
        "execution success receipt differs",
    )

    report_digest, _, report_payload, _ = _stable_file(
        verified / "verification_report.json", keep_payload=True
    )
    assert report_payload is not None
    _require(
        report_digest == verification.report_sha256,
        "independent verification report differs from its receipt",
    )
    verification_report = _json_object(
        report_payload, label="independent verification report"
    )
    _require(
        verification_report.get("status") == VERIFIED_STATUS
        and verification_report.get("verification_id")
        == verification.verification_id
        and verification_report.get("primary_build_id") == publication.build_id
        and verification_report.get("primary_manifest_sha256")
        == publication.manifest_sha256
        and verification_report.get("difference_count") == 0
        and verification_report.get("differences") == [],
        "independent verification did not authenticate this zero-difference build",
    )
    verification_payloads: dict[str, bytes] = {}
    for name in VERIFICATION_MEMBERS:
        digest, _, payload, identity = _stable_file(
            verified / name, keep_payload=True
        )
        assert payload is not None
        identities[verified / name] = identity
        source_hashes[f"verification/{name}"] = digest
        verification_payloads[name] = payload
    differences = _json_object(
        verification_payloads["differences.json"],
        label="independent differences",
    )
    verified_success = _json_object(
        verification_payloads[SUCCESS_NAME],
        label="independent verification success receipt",
    )
    _require(
        differences == {
            "schema_version": 1,
            "difference_count": 0,
            "differences": [],
        },
        "independent differences payload is not empty",
    )
    _require(
        verified_success.get("status") == VERIFIED_STATUS
        and verified_success.get("verification_id") == verification.verification_id
        and verified_success.get("primary_manifest_sha256")
        == publication.manifest_sha256
        and verified_success.get("report_sha256") == verification.report_sha256
        and verified_success.get("differences_sha256")
        == source_hashes["verification/differences.json"]
        and verified_success.get("closed_members") == list(VERIFICATION_MEMBERS),
        "independent verification success receipt differs",
    )

    selection = _json_object(retained[SELECTION_NAME], label="locked selection")
    _require(
        manifest.get("selection_sha256") == selection.get("selection_sha256"),
        "locked selection differs from the execution manifest",
    )
    try:
        monthly = pq.read_table(pa.BufferReader(retained[MONTHLY_METRICS_NAME])).to_pandas()
    except (OSError, pa.ArrowException) as exc:
        raise Stage11ExecutionReportingError(
            "cannot read authenticated monthly metrics"
        ) from exc
    _require(not monthly.empty, "authenticated monthly metrics are empty")
    _require(
        "holding_month" in monthly.columns
        and "strategy_id" in monthly.columns
        and "exit_date" in monthly.columns,
        "authenticated monthly metrics schema differs",
    )
    try:
        exit_dates = pd.to_datetime(monthly["exit_date"], errors="raise")
    except (TypeError, ValueError) as exc:
        raise Stage11ExecutionReportingError(
            "authenticated monthly metric dates are invalid"
        ) from exc
    _require(
        exit_dates.max() <= _MAX_STAGE11_DATE
        and tuple(sorted(set(monthly["holding_month"].astype(str))))
        == VALIDATION_HOLDING_MONTHS,
        "monthly metrics crossed the Stage 11 validation boundary",
    )
    snapshot = _AuthoritySnapshot(
        primary_bundle=primary,
        verification_bundle=verified,
        identities=identities,
        primary_manifest_sha256=publication.manifest_sha256,
        verification_report_sha256=verification.report_sha256,
    )
    snapshot.gate()
    return _AuthenticatedExecutionResults(
        manifest=manifest,
        selection=selection,
        monthly_metrics=monthly,
        verification_report=verification_report,
        source_member_sha256=source_hashes,
        snapshot=snapshot,
    )


def _finite_numeric(frame: pd.DataFrame, name: str) -> np.ndarray:
    _require(name in frame.columns, f"monthly metrics omitted {name}")
    try:
        values = pd.to_numeric(frame[name], errors="raise").to_numpy(np.float64)
    except (TypeError, ValueError) as exc:
        raise Stage11ExecutionReportingError(
            f"monthly metric is not numeric: {name}"
        ) from exc
    _require(np.isfinite(values).all(), f"monthly metric is not finite: {name}")
    return values


def _strategy_metrics(monthly: pd.DataFrame) -> list[dict[str, Any]]:
    required = (
        "pre_trade_nav_dollars",
        "closing_nav_dollars",
        "net_total_return",
        "net_excess_return",
        "transaction_cost_dollars",
        "borrow_cost_dollars",
        "financing_cost_dollars",
        "turnover",
        "gross_exposure",
        "net_exposure",
        "position_count",
    )
    order = {
        PRIMARY_STRATEGY_ID: 0,
        ELASTIC_NET_CHALLENGER_ID: 1,
        RANK_CHALLENGER_ID: 2,
    }
    records: list[dict[str, Any]] = []
    grouped = monthly.assign(
        holding_month=monthly["holding_month"].astype(str)
    ).groupby("strategy_id", sort=False, observed=True)
    for strategy_id, raw_group in grouped:
        group = raw_group.sort_values("holding_month", kind="mergesort")
        months = tuple(group["holding_month"])
        _require(
            months == VALIDATION_HOLDING_MONTHS,
            f"monthly result coverage differs for {strategy_id}",
        )
        values = {name: _finite_numeric(group, name) for name in required}
        _require(
            np.all(values["pre_trade_nav_dollars"] > 0.0)
            and np.all(values["closing_nav_dollars"] > 0.0),
            f"nonpositive NAV in monthly results for {strategy_id}",
        )
        excess = values["net_excess_return"]
        volatility = float(np.std(excess, ddof=1) * math.sqrt(12.0))
        annualized_excess = float(np.mean(excess) * 12.0)
        sharpe = None if volatility == 0.0 else annualized_excess / volatility
        records.append(
            {
                "strategy_id": str(strategy_id),
                "holding_months": len(group),
                "first_holding_month": months[0],
                "last_holding_month": months[-1],
                "initial_nav_dollars": float(values["pre_trade_nav_dollars"][0]),
                "ending_nav_dollars": float(values["closing_nav_dollars"][-1]),
                "cumulative_net_total_return": float(
                    values["closing_nav_dollars"][-1]
                    / values["pre_trade_nav_dollars"][0]
                    - 1.0
                ),
                "annualized_arithmetic_net_excess_return": annualized_excess,
                "annualized_net_excess_volatility": volatility,
                "fully_net_sharpe": sharpe,
                "mean_monthly_net_total_return": float(
                    np.mean(values["net_total_return"])
                ),
                "total_transaction_cost_dollars": float(
                    np.sum(values["transaction_cost_dollars"])
                ),
                "total_borrow_cost_dollars": float(
                    np.sum(values["borrow_cost_dollars"])
                ),
                "total_financing_cost_dollars": float(
                    np.sum(values["financing_cost_dollars"])
                ),
                "mean_monthly_turnover": float(np.mean(values["turnover"])),
                "mean_gross_exposure": float(np.mean(values["gross_exposure"])),
                "mean_net_exposure": float(np.mean(values["net_exposure"])),
                "mean_position_count": float(np.mean(values["position_count"])),
            }
        )
    _require(records, "no strategy metrics were produced")
    return sorted(
        records,
        key=lambda value: (order.get(str(value["strategy_id"]), 99), str(value["strategy_id"])),
    )


def _selection_summary(selection: Mapping[str, Any]) -> dict[str, Any]:
    families = selection.get("families")
    ensemble = selection.get("ensemble")
    _require(
        _is_sha256(selection.get("selection_sha256"))
        and isinstance(families, Mapping)
        and set(families) == {"neural_network", "xgboost", "elastic_net"}
        and isinstance(ensemble, Mapping),
        "locked selection summary schema differs",
    )
    selected_families: dict[str, dict[str, Any]] = {}
    for family in ("neural_network", "xgboost", "elastic_net"):
        raw = families[family]
        _require(isinstance(raw, Mapping), f"selection family differs: {family}")
        selected_families[family] = {
            "config_id": raw.get("config_id"),
            "mean_rank_ic": raw.get("mean_rank_ic"),
            "rank_ic_bootstrap_se": raw.get("rank_ic_bootstrap_se"),
            "signal_turnover": raw.get("signal_turnover"),
            "positive_annual_ic_years": raw.get("positive_annual_ic_years"),
            "annual_ic_gate_pass": raw.get("annual_ic_gate_pass"),
            "mandatory_comparator_retained": raw.get(
                "mandatory_comparator_retained"
            ),
        }
    selected_ensemble = {
        "diagnostic_nn_weight": ensemble.get("diagnostic_nn_weight"),
        "effective_nn_weight": ensemble.get("effective_nn_weight"),
        "effective_xgboost_weight": ensemble.get("effective_xgboost_weight"),
        "mean_rank_ic": ensemble.get("mean_rank_ic"),
        "rank_ic_bootstrap_se": ensemble.get("rank_ic_bootstrap_se"),
        "signal_turnover": ensemble.get("signal_turnover"),
        "forced_by_annual_gate": ensemble.get("forced_by_annual_gate"),
        "active_v1_ml_strategy_qualifies": ensemble.get(
            "active_v1_ml_strategy_qualifies"
        ),
        "gate_reason": ensemble.get("gate_reason"),
    }
    return {
        "selection_sha256": selection["selection_sha256"],
        "families": selected_families,
        "ensemble": selected_ensemble,
    }


def _architecture_summary(manifest: Mapping[str, Any]) -> dict[str, Any]:
    raw = manifest.get("architecture_status")
    _require(isinstance(raw, Mapping), "execution architecture status is missing")
    rows = raw.get("architectures")
    expected_ids = (
        PRIMARY_STRATEGY_ID,
        ELASTIC_NET_CHALLENGER_ID,
        RANK_CHALLENGER_ID,
        DECISION_FOCUSED_CHALLENGER_ID,
    )
    _require(
        raw.get("schema_version") == 1
        and type(raw.get("nonlinear_annual_gate_qualifies")) is bool
        and raw.get("decision_focused_fit_performed") is False
        and isinstance(rows, list)
        and len(rows) == len(expected_ids),
        "execution architecture status schema differs",
    )
    normalized: list[dict[str, Any]] = []
    for expected_id, row in zip(expected_ids, rows, strict=True):
        _require(
            isinstance(row, Mapping)
            and set(row) == {"architecture_id", "active", "inactive_reason"}
            and row.get("architecture_id") == expected_id
            and type(row.get("active")) is bool
            and (
                row.get("inactive_reason") is None
                or isinstance(row.get("inactive_reason"), str)
            ),
            "execution architecture status row differs",
        )
        normalized.append(dict(row))
    qualifies = bool(raw["nonlinear_annual_gate_qualifies"])
    _require(
        normalized[0]["active"] is qualifies
        and normalized[1] == {
            "architecture_id": ELASTIC_NET_CHALLENGER_ID,
            "active": True,
            "inactive_reason": None,
        }
        and normalized[2]["active"] is qualifies
        and normalized[3]["active"] is False
        and normalized[3]["inactive_reason"]
        == (
            "production_training_runtime_not_certified"
            if qualifies
            else "nonlinear_family_gate_failed"
        ),
        "execution architecture activity differs from the locked selection gate",
    )
    return {
        "nonlinear_annual_gate_qualifies": qualifies,
        "decision_focused_fit_performed": False,
        "architectures": normalized,
    }


def _fmt_percent(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 100.0:.2f}%"


def _fmt_number(value: object, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def _markdown(summary: Mapping[str, Any]) -> bytes:
    authority = summary["authority"]
    selection = summary["selection"]
    lines = [
        "# Stage 11 validation results",
        "",
        "## Technical summary",
        "",
        "The 2015–2019 Stage 11 validation execution completed and its existing ",
        f"independent verification reported **{authority['difference_count']} differences**. ",
        "These are validation-selection diagnostics, not the untouched 2020–2025 ",
        "confirmatory out-of-sample result.",
        "",
        "## Verified net strategy results",
        "",
        "| Strategy | Months | Cumulative net return | Annualized net excess return | Annualized volatility | Fully net Sharpe | Mean net exposure |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["strategy_metrics"]:
        lines.append(
            "| "
            + " | ".join(
                (
                    str(row["strategy_id"]),
                    str(row["holding_months"]),
                    _fmt_percent(row["cumulative_net_total_return"]),
                    _fmt_percent(row["annualized_arithmetic_net_excess_return"]),
                    _fmt_percent(row["annualized_net_excess_volatility"]),
                    _fmt_number(row["fully_net_sharpe"]),
                    _fmt_percent(row["mean_net_exposure"]),
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "The fully net Sharpe uses the 59 published calendar-month net excess ",
            "returns. Transaction, short-borrow, and financing costs are already ",
            "included in the published net returns.",
            "",
            "## Locked selector choices",
            "",
            "| Family | Selected configuration | Mean rank IC | Bootstrap SE | Annual IC gate |",
            "|---|---|---:|---:|---|",
        ]
    )
    for family in ("neural_network", "xgboost", "elastic_net"):
        row = selection["families"][family]
        lines.append(
            "| "
            + " | ".join(
                (
                    family,
                    str(row["config_id"]),
                    _fmt_number(row["mean_rank_ic"], 4),
                    _fmt_number(row["rank_ic_bootstrap_se"], 4),
                    str(row["annual_ic_gate_pass"]),
                )
            )
            + " |"
        )
    ensemble = selection["ensemble"]
    lines.extend(
        [
            "",
            f"The diagnostic NN weight is **{_fmt_number(ensemble['diagnostic_nn_weight'], 2)}**; ",
            f"the effective NN weight is **{_fmt_number(ensemble['effective_nn_weight'], 2)}**. ",
            f"The active ML strategy qualification is **{ensemble['active_v1_ml_strategy_qualifies']}** ",
            f"(`{ensemble['gate_reason']}`).",
            "",
            "## Allocation architecture status",
            "",
            "| Architecture ID | Active | Inactive reason |",
            "|---|---|---|",
        ]
    )
    for row in summary["architecture_status"]["architectures"]:
        lines.append(
            f"| {row['architecture_id']} | {row['active']} | "
            f"{row['inactive_reason'] or '—'} |"
        )
    lines.extend(
        [
            "",
            "`decision_focused_fit_performed` is ",
            f"**{summary['architecture_status']['decision_focused_fit_performed']}**; ",
            "no decision-focused model fit was run.",
            "",
            "## Scope, definitions, and limitations",
            "",
            f"- Holding months: **{summary['scope']['first_holding_month']}** through **{summary['scope']['last_holding_month']}** ({summary['scope']['holding_months']} complete months).",
            "- December 2019 is embargoed and excluded; all 2020+ data remain prohibited.",
            "- Annualized arithmetic net excess return is 12 times the mean monthly net excess return.",
            "- Annualized volatility is the sample standard deviation of monthly net excess returns times the square root of 12.",
            "- Fully net Sharpe is annualized arithmetic net excess return divided by annualized volatility; it is `n/a` when volatility is zero.",
            "- No factor alpha, beta, benchmark series, bootstrap confidence interval, or confirmatory inference is added here because those statistics are not members of the verified Stage 11 execution bundle.",
            "",
            "## Exact source authority",
            "",
            f"- Primary build: `{authority['primary_build_id']}` — `{authority['primary_bundle_path']}`",
            f"- Primary manifest SHA-256: `{authority['primary_manifest_sha256']}`",
            f"- Independent verification: `{authority['verification_id']}` — `{authority['verification_bundle_path']}`",
            f"- Verification report SHA-256: `{authority['verification_report_sha256']}`",
            f"- Selection SHA-256: `{selection['selection_sha256']}`",
            "",
            "## Next step",
            "",
            "Treat these results only as the frozen 2015–2019 model-selection evidence. ",
            "Keep 2020–2025 untouched until the separately authorized confirmatory stage.",
            "",
        ]
    )
    return "\n".join(lines).encode("utf-8")


def _report_payloads(
    authenticated: _AuthenticatedExecutionResults,
    publication: Stage11ExecutionPublication,
    verification: Stage11ExecutionVerification,
) -> tuple[str, dict[str, bytes], dict[str, Any]]:
    strategy_metrics = _strategy_metrics(authenticated.monthly_metrics)
    selection = _selection_summary(authenticated.selection)
    architecture_status = _architecture_summary(authenticated.manifest)
    _require(
        architecture_status["nonlinear_annual_gate_qualifies"]
        is selection["ensemble"]["active_v1_ml_strategy_qualifies"],
        "architecture status differs from the locked nonlinear gate",
    )
    manifest_scope = authenticated.manifest["scope"]
    authority_display = {
        "primary_bundle_path": publication.bundle_path.as_posix(),
        "verification_bundle_path": verification.bundle_path.as_posix(),
    }
    identity = {
        "schema_version": 1,
        "publication_id": REPORT_PUBLICATION_ID,
        "status": REPORT_STATUS,
        "classification": (
            "VALIDATION_SELECTION_DIAGNOSTICS_NOT_CONFIRMATORY_OUT_OF_SAMPLE"
        ),
        "scope": {
            "first_holding_month": manifest_scope["first_holding_month"],
            "last_holding_month": manifest_scope["last_holding_month"],
            "holding_months": manifest_scope["holding_months"],
            "december_2019_holding_month_included": False,
            "final_2020_plus_included": False,
        },
        "authority": {
            "primary_build_id": publication.build_id,
            "primary_manifest_sha256": publication.manifest_sha256,
            "primary_closed_member_count": len(PRIMARY_MEMBERS),
            "verification_id": verification.verification_id,
            "verification_report_sha256": verification.report_sha256,
            "verification_closed_member_count": len(VERIFICATION_MEMBERS),
            "difference_count": verification.difference_count,
        },
        "selection": selection,
        "architecture_status": architecture_status,
        "strategy_metrics": strategy_metrics,
        "metric_definitions": {
            "cumulative_net_total_return": (
                "ending_nav_dollars / initial_nav_dollars - 1"
            ),
            "annualized_arithmetic_net_excess_return": (
                "12 * mean(monthly net_excess_return)"
            ),
            "annualized_net_excess_volatility": (
                "sqrt(12) * sample_stddev(monthly net_excess_return, ddof=1)"
            ),
            "fully_net_sharpe": (
                "annualized_arithmetic_net_excess_return / "
                "annualized_net_excess_volatility"
            ),
        },
        "source_members": dict(sorted(authenticated.source_member_sha256.items())),
        "omitted_unproduced_statistics": [
            "factor_alpha",
            "factor_beta",
            "benchmark_returns",
            "bootstrap_confidence_intervals",
            "confirmatory_hypothesis_tests",
        ],
    }
    fingerprint = _sha_bytes(_canonical_json(identity))
    report_id = f"stage11-report-{fingerprint[:24]}"
    summary = {
        **identity,
        "authority": {**identity["authority"], **authority_display},
        "report_id": report_id,
    }
    summary_payload = _canonical_json(summary)
    markdown_payload = _markdown(summary)
    success = {
        "schema_version": 1,
        "publication_id": REPORT_PUBLICATION_ID,
        "status": REPORT_STATUS,
        "report_id": report_id,
        "primary_build_id": publication.build_id,
        "primary_manifest_sha256": publication.manifest_sha256,
        "verification_id": verification.verification_id,
        "verification_report_sha256": verification.report_sha256,
        "difference_count": 0,
        "summary_sha256": _sha_bytes(summary_payload),
        "markdown_sha256": _sha_bytes(markdown_payload),
        "closed_members": list(REPORT_MEMBERS),
    }
    payloads = {
        REPORT_SUMMARY_NAME: summary_payload,
        REPORT_MARKDOWN_NAME: markdown_payload,
        REPORT_SUCCESS_NAME: _canonical_json(success),
    }
    return report_id, payloads, summary


def _report_bundle_gate(bundle: Path, expected_hashes: Mapping[str, str]) -> None:
    _require(
        bundle.is_dir()
        and not bundle.is_symlink()
        and frozenset(path.name for path in bundle.iterdir())
        == frozenset(REPORT_MEMBERS),
        "verified execution report bundle is not closed",
    )
    for name, expected in expected_hashes.items():
        digest, _, _, _ = _stable_file(bundle / name)
        _require(digest == expected, f"verified execution report changed: {name}")


def publish_verified_stage11_execution_report_core(
    publication: Stage11ExecutionPublication,
    verification: Stage11ExecutionVerification,
    *,
    artifact_root: Path,
    expected_synthetic_fixture: bool,
    authority_gate: Callable[[], None] | None = None,
) -> Stage11VerifiedExecutionReportPublication:
    """Publish a verified summary bundle, then its repository pointer last."""

    _require(
        authority_gate is None or callable(authority_gate),
        "verified execution report authority gate must be callable",
    )
    if authority_gate is not None:
        authority_gate()
    authenticated = _authenticate_verified_results(
        publication,
        verification,
        expected_synthetic_fixture=expected_synthetic_fixture,
    )
    report_id, payloads, _ = _report_payloads(
        authenticated, publication, verification
    )
    root = Path(os.path.abspath(artifact_root.expanduser()))
    _require(root.is_absolute(), "artifact root must be absolute")
    root.mkdir(parents=True, exist_ok=True)
    _require(root.is_dir() and not root.is_symlink(), "artifact root is unsafe")
    destination = root / DEFAULT_REPORT_DIRECTORY / report_id
    hashes = {name: _sha_bytes(payload) for name, payload in payloads.items()}

    def source_gate() -> None:
        if authority_gate is not None:
            authority_gate()
        authenticated.snapshot.gate()

    try:
        with SecureStagingDirectory.create(
            trusted_root=root,
            destination=destination,
            expected_files=REPORT_MEMBERS,
            label="Stage 11 verified execution report",
        ) as stage:
            for name in REPORT_MEMBERS:
                with stage.open_exclusive_binary(name) as stream:
                    written = stream.write(payloads[name])
                    _require(
                        written == len(payloads[name]),
                        f"short verified report write: {name}",
                    )
            snapshot = stage.snapshot(expected_hashes=hashes)
            published = stage.publish(
                expected_hashes=snapshot.hashes,
                pre_publish_gate=source_gate,
                post_publish_gate=source_gate,
            )
    except SecurePublicationError as exc:
        raise Stage11ExecutionReportingError(
            "cannot atomically publish verified Stage 11 report"
        ) from exc

    pointer = {
        "schema_version": 1,
        "publication_id": REPORT_PUBLICATION_ID,
        "status": VERIFIED_STATUS,
        "report_id": report_id,
        "report_bundle_path": destination.relative_to(root).as_posix(),
        "report_members": {
            name: {"sha256": hashes[name], "size_bytes": len(payloads[name])}
            for name in REPORT_MEMBERS
        },
        "primary_build_id": publication.build_id,
        "primary_bundle_path": publication.bundle_path.as_posix(),
        "primary_manifest_sha256": publication.manifest_sha256,
        "primary_closed_member_count": len(PRIMARY_MEMBERS),
        "verification_id": verification.verification_id,
        "verification_bundle_path": verification.bundle_path.as_posix(),
        "verification_report_sha256": verification.report_sha256,
        "verification_closed_member_count": len(VERIFICATION_MEMBERS),
        "difference_count": 0,
        "authorizations": {
            "stage11_validation_results_verified": True,
            "stage11_validation_report_is_canonical": True,
            "final_2020_plus_access_authorized": False,
            "stage12_or_later_authorized": False,
        },
    }
    pointer_payload = _canonical_json(pointer)
    pointer_path = root / DEFAULT_LATEST_VERIFIED_POINTER_NAME

    def pointer_gate() -> None:
        source_gate()
        _report_bundle_gate(destination, hashes)

    try:
        replace_pointer_last(
            pointer_path,
            pointer_payload,
            trusted_root=root,
            pre_replace_gate=pointer_gate,
            post_replace_gate=pointer_gate,
            label="Stage 11 latest verified execution pointer",
        )
    except SecurePublicationError as exc:
        raise Stage11ExecutionReportingError(
            "cannot atomically replace latest verified execution pointer"
        ) from exc
    pointer_gate()
    return Stage11VerifiedExecutionReportPublication(
        report_id=report_id,
        bundle_path=destination,
        summary_sha256=hashes[REPORT_SUMMARY_NAME],
        markdown_sha256=hashes[REPORT_MARKDOWN_NAME],
        receipt_sha256=hashes[REPORT_SUCCESS_NAME],
        pointer_path=pointer_path,
        pointer_sha256=_sha_bytes(pointer_payload),
        reused=published.reused_existing,
    )


def publish_verified_stage11_execution_report(
    authority: object,
    publication: Stage11ExecutionPublication,
    verification: Stage11ExecutionVerification,
    *,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
) -> Stage11VerifiedExecutionReportPublication:
    """Authority-bound empirical wrapper for the post-verification report."""

    from portfolio_bot.validation.validation_runner import (
        require_live_stage11_validation_unblind_authority,
    )

    require_live_stage11_validation_unblind_authority(authority)

    def authority_gate() -> None:
        require_live_stage11_validation_unblind_authority(authority)

    result = publish_verified_stage11_execution_report_core(
        publication,
        verification,
        artifact_root=artifact_root,
        expected_synthetic_fixture=False,
        authority_gate=authority_gate,
    )
    require_live_stage11_validation_unblind_authority(authority)
    return result


__all__ = [
    "DEFAULT_ARTIFACT_ROOT",
    "DEFAULT_LATEST_VERIFIED_POINTER_NAME",
    "DEFAULT_REPORT_DIRECTORY",
    "REPORT_MARKDOWN_NAME",
    "REPORT_MEMBERS",
    "REPORT_PUBLICATION_ID",
    "REPORT_STATUS",
    "REPORT_SUCCESS_NAME",
    "REPORT_SUMMARY_NAME",
    "Stage11ExecutionReportingError",
    "Stage11VerifiedExecutionReportPublication",
    "publish_verified_stage11_execution_report",
    "publish_verified_stage11_execution_report_core",
]
