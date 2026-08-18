"""Private authenticated checkpoint for a completed Stage 11 clean portfolio.

The checkpoint exists only to avoid repeating the expensive 59-month
accounting path when a later, non-economic attestation or publication check
fails.  It is not a public result.  Every member is private, content
addressed, independently reopened, and bound to an explicit economic
compatibility payload supplied by the Stage 11 resume orchestrator.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import shutil
import stat
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterator, Mapping

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from portfolio_bot.validation.empirical_portfolio_backend import (
    EmpiricalMarkContinuityEvidence,
    EmpiricalPendingDustRetirementEvidence,
    EmpiricalPortfolioRun,
    EmpiricalRetiredIdentityEvidence,
)
from portfolio_bot.validation.execution_orchestrator import (
    canonical_frame_sha256,
)
from portfolio_bot.validation.execution_publication import (
    DAILY_ACCOUNTING_NAME,
    EXECUTION_SESSIONS_NAME,
    MONTHLY_METRICS_NAME,
    TABLE_COLUMNS,
    TRADES_NAME,
    WEIGHTS_NAME,
)


class Stage11CleanCandidateError(RuntimeError):
    """Raised before an unauthenticated clean candidate can be reused."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Stage11CleanCandidateError(message)


SCHEMA_VERSION = 1
STATUS = "AUTHENTICATED_STAGE11_CLEAN_PORTFOLIO_CANDIDATE"
MANIFEST_NAME = "manifest.json"
EVIDENCE_NAME = "private_evidence.json"
SUCCESS_NAME = "_SUCCESS.json"
DIRECTORY_PREFIX = "stage11-clean-portfolio-"
_TABLE_ATTRIBUTE_BY_NAME = MappingProxyType(
    {
        EXECUTION_SESSIONS_NAME: "execution_sessions",
        WEIGHTS_NAME: "portfolio_weights",
        TRADES_NAME: "portfolio_trades",
        DAILY_ACCOUNTING_NAME: "daily_accounting",
        MONTHLY_METRICS_NAME: "monthly_metrics",
    }
)
MEMBER_NAMES = (
    *_TABLE_ATTRIBUTE_BY_NAME,
    EVIDENCE_NAME,
    MANIFEST_NAME,
    SUCCESS_NAME,
)


@dataclass(frozen=True, slots=True)
class AuthenticatedStage11CleanPortfolioCandidate:
    """Independently reopened private clean-portfolio checkpoint."""

    portfolio: EmpiricalPortfolioRun
    missing_return_projection: Mapping[str, object]
    portfolio_recheck_reuse: Mapping[str, object]
    access_ledger_before_portfolio: Mapping[str, object]
    access_ledger_after_portfolio: Mapping[str, object]
    compatibility: Mapping[str, object]
    table_logical_sha256: Mapping[str, str]
    candidate_sha256: str
    manifest_sha256: str
    bundle_path: Path
    reused: bool


def _canonical_json(value: object) -> bytes:
    try:
        return (
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise Stage11CleanCandidateError(
            "clean candidate contains a non-canonical JSON value"
        ) from exc


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {
            str(key): _plain(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, tuple | list):
        return [_plain(item) for item in value]
    _require(
        value is None or type(value) in {str, int, bool, float},
        "clean candidate contains a non-JSON value",
    )
    return value


def _safe_directory(path: Path, *, create: bool) -> Path:
    _require(path.is_absolute(), "clean candidate root must be absolute")
    lexical = Path(os.path.abspath(path))
    cursor = Path(lexical.anchor)
    for part in lexical.parts[1:]:
        cursor /= part
        if cursor.exists() or cursor.is_symlink():
            metadata = os.lstat(cursor)
            _require(
                not stat.S_ISLNK(metadata.st_mode),
                "clean candidate path contains a symlink",
            )
    if create:
        lexical.mkdir(parents=True, exist_ok=True, mode=0o700)
    _require(
        lexical.is_dir() and not lexical.is_symlink(),
        "clean candidate root is not a safe directory",
    )
    metadata = lexical.stat(follow_symlinks=False)
    _require(
        metadata.st_uid == os.getuid()
        and metadata.st_nlink >= 1
        and stat.S_IMODE(metadata.st_mode) == 0o700,
        "clean candidate directory ownership or mode differs",
    )
    return lexical


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _stable_private_bytes(path: Path, *, label: str) -> bytes:
    before = path.stat(follow_symlinks=False)
    _require(
        stat.S_ISREG(before.st_mode)
        and not path.is_symlink()
        and before.st_uid == os.getuid()
        and before.st_nlink == 1
        and stat.S_IMODE(before.st_mode) == 0o600,
        f"{label} ownership or mode differs",
    )
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        _require(
            (opened.st_dev, opened.st_ino) == (before.st_dev, before.st_ino),
            f"{label} changed before authenticated reopen",
        )
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 8 << 20):
            chunks.append(chunk)
    finally:
        os.close(descriptor)
    after = path.stat(follow_symlinks=False)
    identity = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    _require(
        tuple(getattr(before, name) for name in identity)
        == tuple(getattr(after, name) for name in identity),
        f"{label} changed during authenticated reopen",
    )
    return b"".join(chunks)


def _write_private_file(path: Path, payload: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            written = stream.write(payload)
            _require(written == len(payload), "clean candidate write was short")
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        if path.exists() and not path.is_symlink():
            path.unlink()
        raise


def _parquet_bytes(frame: pd.DataFrame) -> bytes:
    table = pa.Table.from_pandas(frame, preserve_index=False)
    sink = pa.BufferOutputStream()
    pq.write_table(
        table,
        sink,
        compression="zstd",
        compression_level=9,
        use_dictionary=False,
        write_statistics=True,
        row_group_size=100_000,
        data_page_version="1.0",
    )
    return sink.getvalue().to_pybytes()


def _frame_from_parquet(payload: bytes, *, label: str) -> pd.DataFrame:
    try:
        return pq.read_table(pa.BufferReader(payload)).to_pandas()
    except (pa.ArrowException, OSError, ValueError) as exc:
        raise Stage11CleanCandidateError(
            f"{label} is not authenticated Parquet"
        ) from exc


def _pending_dust_payload(
    value: EmpiricalPendingDustRetirementEvidence,
) -> dict[str, object]:
    return {
        "policy_id": value.policy_id,
        "strategy_id": value.strategy_id,
        "holding_month": value.holding_month,
        "boundary_date": value.boundary_date.isoformat(),
        "permno": value.permno,
        "position_value_dollars_hex": value.position_value_dollars.hex(),
        "threshold_dollars_hex": value.threshold_dollars.hex(),
        "last_authoritative_mark_date": (
            value.last_authoritative_mark_date.isoformat()
        ),
        "pending_no_mark_count": value.pending_no_mark_count,
        "pending_evidence_sha256": value.pending_evidence_sha256,
        "opening_portfolio_state_sha256": (
            value.opening_portfolio_state_sha256
        ),
        "closing_portfolio_state_sha256": (
            value.closing_portfolio_state_sha256
        ),
        "opening_mark_ledger_sha256": value.opening_mark_ledger_sha256,
        "closing_mark_ledger_sha256": value.closing_mark_ledger_sha256,
        "opening_nav_dollars_hex": value.opening_nav_dollars.hex(),
        "closing_nav_dollars_hex": value.closing_nav_dollars.hex(),
        "opening_cash_dollars_hex": value.opening_cash_dollars.hex(),
        "closing_cash_dollars_hex": value.closing_cash_dollars.hex(),
        "cash_adjustment_dollars_hex": value.cash_adjustment_dollars.hex(),
        "cash_rounding_residual_dollars_hex": (
            value.cash_rounding_residual_dollars.hex()
        ),
        "execution_price_read": value.execution_price_read,
        "return_imputed": value.return_imputed,
        "trade_emitted": value.trade_emitted,
        "transaction_cost_imputed": value.transaction_cost_imputed,
        "corporate_action_applied": value.corporate_action_applied,
        "identity_terminally_retired": value.identity_terminally_retired,
    }


def _evidence_payload(
    *,
    portfolio: EmpiricalPortfolioRun,
    missing_return_projection: Mapping[str, object],
    portfolio_recheck_reuse: Mapping[str, object],
    access_ledger_before_portfolio: Mapping[str, object],
    access_ledger_after_portfolio: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "execution_session_receipts": _plain(
            portfolio.execution_session_receipts
        ),
        "mark_continuity_evidence": [
            {
                "strategy_id": value.strategy_id,
                "holding_month": value.holding_month,
                "trading_date": value.trading_date.isoformat(),
                "opening_ledger_sha256": value.opening_ledger_sha256,
                "closing_ledger_sha256": value.closing_ledger_sha256,
                "no_mark_permnos": list(value.no_mark_permnos),
                "resolved_pending_permnos": list(
                    value.resolved_pending_permnos
                ),
                "observed_zero_return_permnos": list(
                    value.observed_zero_return_permnos
                ),
            }
            for value in portfolio.mark_continuity_evidence
        ],
        "retired_identity_evidence": [
            {
                "holding_month": value.holding_month,
                "signal_date": value.signal_date.isoformat(),
                "entry_date": value.entry_date.isoformat(),
                "permno": value.permno,
                "retirement_date": value.retirement_date.isoformat(),
                "prior_long_entry_allowed": value.prior_long_entry_allowed,
                "prior_short_entry_allowed": value.prior_short_entry_allowed,
            }
            for value in portfolio.retired_identity_evidence
        ],
        "pending_dust_retirement_evidence": [
            _pending_dust_payload(value)
            for value in portfolio.pending_dust_retirement_evidence
        ],
        "final_mark_continuity_ledger_sha256": _plain(
            portfolio.final_mark_continuity_ledger_sha256
        ),
        "final_pending_permnos": {
            str(strategy_id): list(values)
            for strategy_id, values in sorted(
                portfolio.final_pending_permnos.items()
            )
        },
        "final_pending_nonzero_count": portfolio.final_pending_nonzero_count,
        "missing_return_projection": _plain(missing_return_projection),
        "portfolio_recheck_reuse": _plain(portfolio_recheck_reuse),
        "access_ledger_before_portfolio": _plain(
            access_ledger_before_portfolio
        ),
        "access_ledger_after_portfolio": _plain(
            access_ledger_after_portfolio
        ),
    }


def _parse_date(value: object, *, label: str) -> dt.date:
    _require(isinstance(value, str), f"{label} is not an ISO date")
    try:
        return dt.date.fromisoformat(value)
    except ValueError as exc:
        raise Stage11CleanCandidateError(f"{label} is not an ISO date") from exc


def _parse_float_hex(value: object, *, label: str) -> float:
    _require(isinstance(value, str), f"{label} is not an exact float")
    try:
        result = float.fromhex(value)
    except ValueError as exc:
        raise Stage11CleanCandidateError(
            f"{label} is not an exact float"
        ) from exc
    _require(result.hex() == value, f"{label} float encoding is non-canonical")
    return result


def _mapping(value: object, *, label: str) -> dict[str, Any]:
    _require(isinstance(value, Mapping), f"{label} is not a mapping")
    return {str(key): item for key, item in value.items()}


@contextmanager
def _authenticated_pending_dust_reopen_scope(
    *,
    evidence: Mapping[str, Any],
    compatibility: Mapping[str, object],
) -> Iterator[None]:
    """Reconstruct amended evidence under its authenticated Stage 11 policy."""

    raw_rows = evidence.get("pending_dust_retirement_evidence")
    _require(
        isinstance(raw_rows, list),
        "clean candidate pending-dust evidence is not a list",
    )
    if not raw_rows:
        yield
        return

    from stage_11_pending_materiality_adapter import (
        PENDING_MATERIALITY_POLICY_ID,
        PENDING_MATERIALITY_THRESHOLD_DOLLARS,
        stage11_pending_materiality_scope,
    )

    policy = _mapping(
        compatibility.get("pending_identity_materiality_policy"),
        label="clean candidate pending-dust policy",
    )
    expected_threshold_hex = (
        PENDING_MATERIALITY_THRESHOLD_DOLLARS.hex()
    )
    _require(
        policy.get("policy_id") == PENDING_MATERIALITY_POLICY_ID
        and policy.get(
            "strict_absolute_position_value_dollars_upper_bound"
        )
        == PENDING_MATERIALITY_THRESHOLD_DOLLARS
        and all(
            isinstance(raw, Mapping)
            and raw.get("policy_id") == PENDING_MATERIALITY_POLICY_ID
            and raw.get("threshold_dollars_hex")
            == expected_threshold_hex
            for raw in raw_rows
        ),
        "clean candidate pending-dust policy binding differs",
    )
    with stage11_pending_materiality_scope():
        yield


def _portfolio_from_payload(
    *,
    frames: Mapping[str, pd.DataFrame],
    evidence: Mapping[str, Any],
) -> EmpiricalPortfolioRun:
    receipts = {
        str(month): MappingProxyType(
            {
                str(key): str(value)
                for key, value in sorted(
                    _mapping(raw_receipt, label="execution session receipt").items()
                )
            }
        )
        for month, raw_receipt in sorted(
            _mapping(
                evidence.get("execution_session_receipts"),
                label="execution session receipts",
            ).items()
        )
    }
    continuity = tuple(
        EmpiricalMarkContinuityEvidence(
            strategy_id=str(raw["strategy_id"]),
            holding_month=str(raw["holding_month"]),
            trading_date=_parse_date(
                raw["trading_date"], label="mark-continuity trading date"
            ),
            opening_ledger_sha256=str(raw["opening_ledger_sha256"]),
            closing_ledger_sha256=str(raw["closing_ledger_sha256"]),
            no_mark_permnos=tuple(int(value) for value in raw["no_mark_permnos"]),
            resolved_pending_permnos=tuple(
                int(value) for value in raw["resolved_pending_permnos"]
            ),
            observed_zero_return_permnos=tuple(
                int(value) for value in raw["observed_zero_return_permnos"]
            ),
        )
        for raw in evidence.get("mark_continuity_evidence", ())
    )
    retired = tuple(
        EmpiricalRetiredIdentityEvidence(
            holding_month=str(raw["holding_month"]),
            signal_date=_parse_date(
                raw["signal_date"], label="retired signal date"
            ),
            entry_date=_parse_date(raw["entry_date"], label="retired entry date"),
            permno=int(raw["permno"]),
            retirement_date=_parse_date(
                raw["retirement_date"], label="identity retirement date"
            ),
            prior_long_entry_allowed=raw["prior_long_entry_allowed"],
            prior_short_entry_allowed=raw["prior_short_entry_allowed"],
        )
        for raw in evidence.get("retired_identity_evidence", ())
    )
    pending = tuple(
        EmpiricalPendingDustRetirementEvidence(
            policy_id=str(raw["policy_id"]),
            strategy_id=str(raw["strategy_id"]),
            holding_month=str(raw["holding_month"]),
            boundary_date=_parse_date(
                raw["boundary_date"], label="pending-dust boundary date"
            ),
            permno=int(raw["permno"]),
            position_value_dollars=_parse_float_hex(
                raw["position_value_dollars_hex"],
                label="pending-dust position value",
            ),
            threshold_dollars=_parse_float_hex(
                raw["threshold_dollars_hex"],
                label="pending-dust threshold",
            ),
            last_authoritative_mark_date=_parse_date(
                raw["last_authoritative_mark_date"],
                label="pending-dust last mark date",
            ),
            pending_no_mark_count=int(raw["pending_no_mark_count"]),
            pending_evidence_sha256=str(raw["pending_evidence_sha256"]),
            opening_portfolio_state_sha256=str(
                raw["opening_portfolio_state_sha256"]
            ),
            closing_portfolio_state_sha256=str(
                raw["closing_portfolio_state_sha256"]
            ),
            opening_mark_ledger_sha256=str(
                raw["opening_mark_ledger_sha256"]
            ),
            closing_mark_ledger_sha256=str(
                raw["closing_mark_ledger_sha256"]
            ),
            opening_nav_dollars=_parse_float_hex(
                raw["opening_nav_dollars_hex"],
                label="pending-dust opening NAV",
            ),
            closing_nav_dollars=_parse_float_hex(
                raw["closing_nav_dollars_hex"],
                label="pending-dust closing NAV",
            ),
            opening_cash_dollars=_parse_float_hex(
                raw["opening_cash_dollars_hex"],
                label="pending-dust opening cash",
            ),
            closing_cash_dollars=_parse_float_hex(
                raw["closing_cash_dollars_hex"],
                label="pending-dust closing cash",
            ),
            cash_adjustment_dollars=_parse_float_hex(
                raw["cash_adjustment_dollars_hex"],
                label="pending-dust cash adjustment",
            ),
            cash_rounding_residual_dollars=_parse_float_hex(
                raw["cash_rounding_residual_dollars_hex"],
                label="pending-dust cash residual",
            ),
            execution_price_read=raw["execution_price_read"],
            return_imputed=raw["return_imputed"],
            trade_emitted=raw["trade_emitted"],
            transaction_cost_imputed=raw["transaction_cost_imputed"],
            corporate_action_applied=raw["corporate_action_applied"],
            identity_terminally_retired=raw["identity_terminally_retired"],
        )
        for raw in evidence.get("pending_dust_retirement_evidence", ())
    )
    final_ledgers = MappingProxyType(
        {
            str(key): str(value)
            for key, value in sorted(
                _mapping(
                    evidence.get("final_mark_continuity_ledger_sha256"),
                    label="final mark ledgers",
                ).items()
            )
        }
    )
    final_pending = MappingProxyType(
        {
            str(key): tuple(int(item) for item in value)
            for key, value in sorted(
                _mapping(
                    evidence.get("final_pending_permnos"),
                    label="final pending identities",
                ).items()
            )
        }
    )
    return EmpiricalPortfolioRun(
        execution_sessions=frames[EXECUTION_SESSIONS_NAME],
        execution_session_receipts=MappingProxyType(receipts),
        portfolio_weights=frames[WEIGHTS_NAME],
        portfolio_trades=frames[TRADES_NAME],
        daily_accounting=frames[DAILY_ACCOUNTING_NAME],
        monthly_metrics=frames[MONTHLY_METRICS_NAME],
        mark_continuity_evidence=continuity,
        retired_identity_evidence=retired,
        pending_dust_retirement_evidence=pending,
        final_mark_continuity_ledger_sha256=final_ledgers,
        final_pending_permnos=final_pending,
        final_pending_nonzero_count=int(
            evidence.get("final_pending_nonzero_count")
        ),
    )


def _load_bundle(
    bundle: Path,
    *,
    expected_compatibility: Mapping[str, object] | None,
    reused: bool,
) -> AuthenticatedStage11CleanPortfolioCandidate | None:
    _safe_directory(bundle, create=False)
    _require(
        set(path.name for path in bundle.iterdir()) == set(MEMBER_NAMES),
        "clean candidate member set differs",
    )
    payloads = {
        name: _stable_private_bytes(
            bundle / name, label=f"clean candidate {name}"
        )
        for name in MEMBER_NAMES
    }
    try:
        manifest = json.loads(payloads[MANIFEST_NAME])
        success = json.loads(payloads[SUCCESS_NAME])
        evidence = json.loads(payloads[EVIDENCE_NAME])
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Stage11CleanCandidateError(
            "clean candidate JSON member is invalid"
        ) from exc
    _require(
        isinstance(manifest, Mapping)
        and manifest.get("schema_version") == SCHEMA_VERSION
        and manifest.get("status") == STATUS,
        "clean candidate manifest identity differs",
    )
    identity = dict(manifest)
    candidate_sha256 = str(identity.pop("candidate_sha256", ""))
    _require(
        len(candidate_sha256) == 64
        and _sha256(_canonical_json(identity)) == candidate_sha256
        and bundle.name == f"{DIRECTORY_PREFIX}{candidate_sha256}",
        "clean candidate content address differs",
    )
    manifest_sha256 = _sha256(payloads[MANIFEST_NAME])
    _require(
        success
        == {
            "schema_version": SCHEMA_VERSION,
            "status": STATUS,
            "candidate_sha256": candidate_sha256,
            "manifest_sha256": manifest_sha256,
            "closed_members": list(MEMBER_NAMES),
        },
        "clean candidate success receipt differs",
    )
    members = _mapping(manifest.get("members"), label="candidate members")
    _require(
        set(members) == set(MEMBER_NAMES) - {MANIFEST_NAME, SUCCESS_NAME},
        "clean candidate manifest member set differs",
    )
    for name, receipt_value in members.items():
        receipt = _mapping(receipt_value, label=f"{name} receipt")
        _require(
            receipt
            == {
                "sha256": _sha256(payloads[name]),
                "size_bytes": len(payloads[name]),
            },
            f"clean candidate member receipt differs: {name}",
        )
    compatibility = _mapping(
        manifest.get("compatibility"), label="candidate compatibility"
    )
    _require(
        manifest.get("compatibility_sha256")
        == _sha256(_canonical_json(compatibility)),
        "clean candidate compatibility digest differs",
    )
    if expected_compatibility is not None and compatibility != _plain(
        expected_compatibility
    ):
        return None
    table_receipts = _mapping(
        manifest.get("tables"), label="candidate table receipts"
    )
    _require(
        set(table_receipts) == set(_TABLE_ATTRIBUTE_BY_NAME),
        "clean candidate table receipt set differs",
    )
    frames: dict[str, pd.DataFrame] = {}
    for name in _TABLE_ATTRIBUTE_BY_NAME:
        frame = _frame_from_parquet(payloads[name], label=name)
        receipt = _mapping(table_receipts[name], label=f"{name} logical receipt")
        _require(
            tuple(frame.columns) == TABLE_COLUMNS[name]
            and receipt
            == {
                "columns": list(TABLE_COLUMNS[name]),
                "logical_sha256": canonical_frame_sha256(frame),
                "rows": len(frame),
            },
            f"clean candidate logical table differs: {name}",
        )
        frames[name] = frame
    _require(
        isinstance(evidence, Mapping)
        and evidence.get("schema_version") == SCHEMA_VERSION,
        "clean candidate private evidence identity differs",
    )
    with _authenticated_pending_dust_reopen_scope(
        evidence=evidence,
        compatibility=compatibility,
    ):
        portfolio = _portfolio_from_payload(frames=frames, evidence=evidence)
    continuity_keys = tuple(
        (
            value.strategy_id,
            value.holding_month,
            value.trading_date,
        )
        for value in portfolio.mark_continuity_evidence
    )
    daily = portfolio.daily_accounting
    daily_dates = (
        ()
        if daily.empty
        else tuple(
            pd.to_datetime(daily["date"], errors="raise").dt.date
        )
    )
    daily_keys = tuple(
        zip(
            daily["strategy_id"].astype(str),
            daily["holding_month"].astype(str),
            daily_dates,
            strict=True,
        )
    )
    _require(
        len(continuity_keys) == len(set(continuity_keys))
        and set(continuity_keys) == set(daily_keys)
        and all(
            isinstance(value.strategy_id, str)
            and bool(value.strategy_id)
            and isinstance(value.holding_month, str)
            and bool(value.holding_month)
            and type(value.trading_date) is dt.date
            and all(
                isinstance(digest, str)
                and len(digest) == 64
                and all(character in "0123456789abcdef" for character in digest)
                for digest in (
                    value.opening_ledger_sha256,
                    value.closing_ledger_sha256,
                )
            )
            and all(
                values == tuple(sorted(set(values)))
                and all(type(permno) is int and permno > 0 for permno in values)
                for values in (
                    value.no_mark_permnos,
                    value.resolved_pending_permnos,
                    value.observed_zero_return_permnos,
                )
            )
            for value in portfolio.mark_continuity_evidence
        ),
        "clean candidate mark-continuity evidence differs from daily accounting",
    )
    missing_return_projection = _mapping(
        evidence.get("missing_return_projection"),
        label="missing-return audit",
    )
    portfolio_recheck_reuse = _mapping(
        evidence.get("portfolio_recheck_reuse"),
        label="portfolio recheck audit",
    )
    access_before = _mapping(
        evidence.get("access_ledger_before_portfolio"),
        label="pre-portfolio access ledger",
    )
    access_after = _mapping(
        evidence.get("access_ledger_after_portfolio"),
        label="post-portfolio access ledger",
    )
    _require(
        _evidence_payload(
            portfolio=portfolio,
            missing_return_projection=missing_return_projection,
            portfolio_recheck_reuse=portfolio_recheck_reuse,
            access_ledger_before_portfolio=access_before,
            access_ledger_after_portfolio=access_after,
        )
        == evidence,
        "clean candidate private evidence is not canonical",
    )
    return AuthenticatedStage11CleanPortfolioCandidate(
        portfolio=portfolio,
        missing_return_projection=MappingProxyType(missing_return_projection),
        portfolio_recheck_reuse=MappingProxyType(portfolio_recheck_reuse),
        access_ledger_before_portfolio=MappingProxyType(access_before),
        access_ledger_after_portfolio=MappingProxyType(access_after),
        compatibility=MappingProxyType(compatibility),
        table_logical_sha256=MappingProxyType(
            {
                name: str(
                    _mapping(
                        table_receipts[name],
                        label=f"{name} logical receipt",
                    )["logical_sha256"]
                )
                for name in _TABLE_ATTRIBUTE_BY_NAME
            }
        ),
        candidate_sha256=candidate_sha256,
        manifest_sha256=manifest_sha256,
        bundle_path=bundle,
        reused=reused,
    )


def load_authenticated_stage11_clean_candidate(
    *,
    root: Path,
    compatibility: Mapping[str, object],
) -> AuthenticatedStage11CleanPortfolioCandidate | None:
    """Load the unique exact-compatible candidate, if one exists."""

    if not root.exists() and not root.is_symlink():
        return None
    safe_root = _safe_directory(root, create=False)
    candidates: list[AuthenticatedStage11CleanPortfolioCandidate] = []
    for path in sorted(safe_root.iterdir(), key=lambda value: value.name):
        _require(
            path.name.startswith(DIRECTORY_PREFIX)
            and path.is_dir()
            and not path.is_symlink(),
            "clean candidate root contains an unexpected entry",
        )
        candidate = _load_bundle(
            path,
            expected_compatibility=compatibility,
            reused=True,
        )
        if candidate is not None:
            candidates.append(candidate)
    _require(
        len(candidates) <= 1,
        "multiple compatible clean portfolio candidates exist",
    )
    return None if not candidates else candidates[0]


def publish_authenticated_stage11_clean_candidate(
    *,
    root: Path,
    compatibility: Mapping[str, object],
    portfolio: EmpiricalPortfolioRun,
    missing_return_projection: Mapping[str, object],
    portfolio_recheck_reuse: Mapping[str, object],
    access_ledger_before_portfolio: Mapping[str, object],
    access_ledger_after_portfolio: Mapping[str, object],
) -> AuthenticatedStage11CleanPortfolioCandidate:
    """Atomically persist and independently reopen one clean candidate."""

    _require(
        type(portfolio) is EmpiricalPortfolioRun,
        "clean candidate requires an exact EmpiricalPortfolioRun",
    )
    safe_root = _safe_directory(root, create=True)
    payloads: dict[str, bytes] = {}
    table_receipts: dict[str, dict[str, object]] = {}
    for name, attribute in _TABLE_ATTRIBUTE_BY_NAME.items():
        frame = getattr(portfolio, attribute)
        _require(
            isinstance(frame, pd.DataFrame)
            and set(frame.columns) == set(TABLE_COLUMNS[name]),
            f"clean candidate table schema differs: {name}",
        )
        ordered = frame.loc[:, list(TABLE_COLUMNS[name])].copy(deep=True)
        live_logical_sha256 = canonical_frame_sha256(ordered)
        payload = _parquet_bytes(ordered)
        reopened = _frame_from_parquet(payload, label=name)
        reopened_logical_sha256 = canonical_frame_sha256(reopened)
        _require(
            tuple(reopened.columns) == TABLE_COLUMNS[name]
            and reopened_logical_sha256 == live_logical_sha256,
            f"clean candidate Parquet schema differs: {name}",
        )
        payloads[name] = payload
        table_receipts[name] = {
            "columns": list(TABLE_COLUMNS[name]),
            "logical_sha256": reopened_logical_sha256,
            "rows": len(reopened),
        }
    evidence = _evidence_payload(
        portfolio=portfolio,
        missing_return_projection=missing_return_projection,
        portfolio_recheck_reuse=portfolio_recheck_reuse,
        access_ledger_before_portfolio=access_ledger_before_portfolio,
        access_ledger_after_portfolio=access_ledger_after_portfolio,
    )
    payloads[EVIDENCE_NAME] = _canonical_json(evidence)
    plain_compatibility = _plain(compatibility)
    identity: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "status": STATUS,
        "compatibility": plain_compatibility,
        "compatibility_sha256": _sha256(
            _canonical_json(plain_compatibility)
        ),
        "tables": table_receipts,
        "members": {
            name: {
                "sha256": _sha256(payload),
                "size_bytes": len(payload),
            }
            for name, payload in sorted(payloads.items())
        },
    }
    candidate_sha256 = _sha256(_canonical_json(identity))
    manifest = {**identity, "candidate_sha256": candidate_sha256}
    payloads[MANIFEST_NAME] = _canonical_json(manifest)
    payloads[SUCCESS_NAME] = _canonical_json(
        {
            "schema_version": SCHEMA_VERSION,
            "status": STATUS,
            "candidate_sha256": candidate_sha256,
            "manifest_sha256": _sha256(payloads[MANIFEST_NAME]),
            "closed_members": list(MEMBER_NAMES),
        }
    )
    destination = safe_root / f"{DIRECTORY_PREFIX}{candidate_sha256}"
    if destination.exists() or destination.is_symlink():
        candidate = _load_bundle(
            destination,
            expected_compatibility=compatibility,
            reused=True,
        )
        _require(candidate is not None, "existing clean candidate differs")
        return candidate
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{DIRECTORY_PREFIX}", dir=safe_root)
    )
    try:
        os.chmod(temporary, 0o700)
        for name in MEMBER_NAMES:
            _write_private_file(temporary / name, payloads[name])
        _fsync_directory(temporary)
        os.replace(temporary, destination)
        _fsync_directory(safe_root)
    finally:
        if temporary.exists() and not temporary.is_symlink():
            shutil.rmtree(temporary)
    candidate = _load_bundle(
        destination,
        expected_compatibility=compatibility,
        reused=False,
    )
    _require(candidate is not None, "published clean candidate cannot reopen")
    return candidate


__all__ = [
    "AuthenticatedStage11CleanPortfolioCandidate",
    "Stage11CleanCandidateError",
    "load_authenticated_stage11_clean_candidate",
    "publish_authenticated_stage11_clean_candidate",
]
