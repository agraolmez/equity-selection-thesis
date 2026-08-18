"""Sealed Gate-A authority for Stage 11 validation source materialization.

Gate A is intentionally narrower than the validation-outcome unblind.  It may
date-slice and normalize the fixed market/accounting sources needed by the
already frozen Stage 11 engine, but it cannot construct labels, fit or select
models, form portfolios, create trades, or calculate performance.  Gate A and
the later execution gate use different approval contracts and different live
authority types; neither authority can be substituted for the other.

The repository initially ships with the two approval digests below unset.
Consequently :func:`open_stage11_validation_source_materialization` fails
before it stats the approval, plan, any vendor file, or any output directory.
Only an explicit user approval, recorded verbatim and then reviewed into the
two frozen digest constants, can make this authority live.  The fixed source
plan authenticates this module with a narrowly normalized digest that masks
only those two literal right-hand sides.  This avoids a circular dependency
between the plan, approval, and approval digests while continuing to bind all
executable gate logic byte-for-byte.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterator, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PLAN_RELATIVE_PATH = (
    "configs/phase_05_validation/stage_11_source_materialization_plan_v1.json"
)
APPROVAL_RELATIVE_PATH = (
    "configs/phase_05_validation/stage_11_source_materialization_approval_v1.json"
)
PLAN_PATH = PROJECT_ROOT / PLAN_RELATIVE_PATH
APPROVAL_PATH = PROJECT_ROOT / APPROVAL_RELATIVE_PATH

PLAN_ID = "phase05-stage11-source-materialization-plan-v1"
APPROVAL_CONTRACT_ID = "phase05-stage11-source-materialization-approval-v1"
APPROVAL_SCOPE = "stage11_validation_source_materialization_gate_a"
GATE_A_IMPLEMENTATION_HASH_MODE = "gate_a_approval_rhs_normalized_v1"
_SELF_IMPLEMENTATION_RELATIVE_PATH = (
    "src/portfolio_bot/validation/validation_source_materialization.py"
)
_APPROVAL_DIGEST_CONSTANT_NAMES = (
    "FROZEN_SOURCE_MATERIALIZATION_APPROVAL_FILE_SHA256",
    "FROZEN_SOURCE_MATERIALIZATION_APPROVAL_PAYLOAD_SHA256",
)

# Deliberately unset.  A user approval is not implied by technical readiness.
# After an exact approval contract exists, a reviewed repository change must
# pin both the file digest and its self-authenticated logical-payload digest.
FROZEN_SOURCE_MATERIALIZATION_APPROVAL_FILE_SHA256: str | None = "a28b70f89c5f11a92be4914714b664d3a4feecd52aa38ed24010a2c5872bb5af"
FROZEN_SOURCE_MATERIALIZATION_APPROVAL_PAYLOAD_SHA256: str | None = "59fbf47780eddc390b61f1a05aea313bd191f2ea1114e7026207e9d5aa431115"

MAIN_START = dt.date(2015, 1, 2)
MAIN_END = dt.date(2019, 10, 31)
RESERVE_START = dt.date(2019, 11, 1)
RESERVE_END = dt.date(2019, 12, 2)
FIRST_PROHIBITED_DATE = dt.date(2020, 1, 2)

_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_UTC = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z\Z")
_APPROVAL_FIELDS = frozenset(
    {
        "schema_version",
        "contract_id",
        "stage",
        "gate",
        "scope",
        "approved",
        "approved_by",
        "approved_at_utc",
        "approval_text",
        "approval_text_sha256",
        "source_materialization_plan_sha256",
        "stage10_verified_pointer_sha256",
        "methodology_source_sha256",
        "authorizations",
        "payload_sha256",
    }
)
_EXACT_AUTHORIZATIONS = MappingProxyType(
    {
        "normalized_source_materialization_authorized": True,
        "main_2015_01_02_through_2019_10_31_authorized": True,
        "reserve_2019_11_01_through_2019_12_02_authorized": True,
        "labels_or_targets_authorized": False,
        "model_fit_or_selection_authorized": False,
        "forecasts_authorized": False,
        "portfolio_weights_authorized": False,
        "trades_authorized": False,
        "pnl_or_performance_authorized": False,
        "december_2019_holding_month_authorized": False,
        "final_2020_2025_authorized": False,
        "stage12_or_later_authorized": False,
    }
)


class Stage11SourceMaterializationSealedError(RuntimeError):
    """Raised before any source/output lookup when Gate A is sealed."""


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


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Stage11SourceMaterializationSealedError(message)


def _hash(value: object, label: str) -> str:
    _require(
        isinstance(value, str) and _HEX64.fullmatch(value) is not None,
        f"{label} is not a lowercase SHA-256",
    )
    return value


def _implementation_sha256(relative: str, path: Path) -> str:
    """Hash one implementation, masking only the two reviewed Gate-A digests.

    Every implementation except this authority module uses its raw bytes.  For
    this module, each exact top-level constant assignment must occur once and
    its right-hand side must be either ``None`` or one double-quoted lowercase
    SHA-256 literal.  Only that right-hand-side span is normalized to ``None``.
    Missing, duplicated, dynamic, or differently formatted assignments fail
    closed instead of expanding the normalization surface.
    """

    payload = path.read_bytes()
    if relative != _SELF_IMPLEMENTATION_RELATIVE_PATH:
        return sha256_bytes(payload)
    normalized = payload
    for name in _APPROVAL_DIGEST_CONSTANT_NAMES:
        prefix = f"{name}: str | None = ".encode("ascii")
        pattern = re.compile(
            rb"^"
            + re.escape(prefix)
            + rb'(?:None|"[0-9a-f]{64}")$',
            flags=re.MULTILINE,
        )
        matches = tuple(pattern.finditer(normalized))
        _require(
            len(matches) == 1,
            f"Gate-A approval digest assignment differs: {name}",
        )
        match = matches[0]
        rhs_start = match.start() + len(prefix)
        normalized = normalized[:rhs_start] + b"None" + normalized[match.end() :]
    return sha256_bytes(normalized)


def _stable_regular_bytes(path: Path, *, expected_sha256: str, label: str) -> bytes:
    """Read one ordinary regular file while binding its complete identity."""

    _require(path.is_absolute(), f"{label} path is not absolute")
    lexical = Path(os.path.abspath(path.expanduser()))
    cursor = Path(lexical.anchor)
    for component in lexical.parts[1:]:
        cursor /= component
        try:
            observed = os.lstat(cursor)
        except OSError as exc:
            raise Stage11SourceMaterializationSealedError(
                f"cannot authenticate {label} path"
            ) from exc
        _require(
            not stat.S_ISLNK(observed.st_mode), f"symlinked {label} path is prohibited"
        )
    before = os.lstat(lexical)
    _require(
        stat.S_ISREG(before.st_mode) and before.st_nlink == 1,
        f"{label} is not a single-link regular file",
    )
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(lexical, flags)
    try:
        opened = os.fstat(descriptor)
        identity = lambda value: (
            value.st_dev,
            value.st_ino,
            value.st_mode,
            value.st_nlink,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )
        _require(identity(before) == identity(opened), f"{label} changed while opened")
        chunks: list[bytes] = []
        while block := os.read(descriptor, 1 << 20):
            chunks.append(block)
        payload = b"".join(chunks)
        after = os.fstat(descriptor)
        selected = os.lstat(lexical)
        _require(
            identity(before) == identity(after) == identity(selected),
            f"{label} changed while read",
        )
    finally:
        os.close(descriptor)
    _require(sha256_bytes(payload) == expected_sha256, f"{label} SHA-256 differs")
    return payload


def _approval_payload_without_digest(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value[key] for key in sorted(_APPROVAL_FIELDS - {"payload_sha256"})}


def validate_approval_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate the exact canonical Gate-A approval shape and self-digest."""

    _require(isinstance(value, Mapping), "Gate-A approval is not an object")
    _require(frozenset(value) == _APPROVAL_FIELDS, "Gate-A approval field set differs")
    _require(value["schema_version"] == 1, "Gate-A approval schema version differs")
    _require(
        value["contract_id"] == APPROVAL_CONTRACT_ID, "Gate-A approval contract differs"
    )
    _require(
        value["stage"] == 11 and value["gate"] == "A",
        "Gate-A approval stage/gate differs",
    )
    _require(value["scope"] == APPROVAL_SCOPE, "Gate-A approval scope differs")
    _require(
        value["approved"] is True, "Gate-A source materialization was not approved"
    )
    for name in ("approved_by", "approval_text"):
        _require(
            isinstance(value[name], str)
            and value[name].strip() == value[name]
            and bool(value[name]),
            f"Gate-A {name} is invalid",
        )
    timestamp = value["approved_at_utc"]
    _require(
        isinstance(timestamp, str) and _UTC.fullmatch(timestamp) is not None,
        "Gate-A approval timestamp is invalid",
    )
    try:
        parsed = dt.datetime.fromisoformat(timestamp[:-1] + "+00:00")
    except ValueError as exc:
        raise Stage11SourceMaterializationSealedError(
            "Gate-A approval timestamp is invalid"
        ) from exc
    _require(
        parsed.utcoffset() == dt.timedelta(0), "Gate-A approval timestamp is not UTC"
    )
    _require(
        _hash(value["approval_text_sha256"], "approval_text_sha256")
        == sha256_bytes(value["approval_text"].encode("utf-8")),
        "Gate-A approval text digest differs",
    )
    for name in (
        "source_materialization_plan_sha256",
        "stage10_verified_pointer_sha256",
        "methodology_source_sha256",
        "payload_sha256",
    ):
        _hash(value[name], name)
    _require(
        dict(value["authorizations"]) == dict(_EXACT_AUTHORIZATIONS),
        "Gate-A approval authorizations differ",
    )
    expected = sha256_bytes(
        canonical_json_bytes(_approval_payload_without_digest(value))
    )
    _require(
        value["payload_sha256"] == expected, "Gate-A approval payload digest differs"
    )
    return MappingProxyType(dict(value))


def _validate_plan_shape(value: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate non-negotiable scope and capability limits of the fixed plan."""

    exact = {
        "schema_version",
        "plan_id",
        "stage",
        "gate",
        "scope",
        "upstream_authorities",
        "sources",
        "outputs",
        "publication",
        "implementation_code_hashes",
        "authorizations",
        "payload_sha256",
    }
    _require(
        isinstance(value, Mapping) and set(value) == exact,
        "Gate-A source plan field set differs",
    )
    _require(
        value["schema_version"] == 1 and value["plan_id"] == PLAN_ID,
        "Gate-A source plan identity differs",
    )
    _require(
        value["stage"] == 11 and value["gate"] == "A",
        "Gate-A source plan stage/gate differs",
    )
    _require(
        value["scope"]
        == {
            "main_start": MAIN_START.isoformat(),
            "main_end": MAIN_END.isoformat(),
            "reserve_start": RESERVE_START.isoformat(),
            "reserve_end": RESERVE_END.isoformat(),
            "first_prohibited_date": FIRST_PROHIBITED_DATE.isoformat(),
            "physical_main_reserve_separation_required": True,
        },
        "Gate-A source plan scope differs",
    )
    _require(
        dict(value["authorizations"]) == dict(_EXACT_AUTHORIZATIONS),
        "Gate-A source plan authorizations differ",
    )
    upstream = value["upstream_authorities"]
    _require(
        isinstance(upstream, dict) and bool(upstream),
        "Gate-A upstream authority set is empty",
    )
    for name, record in upstream.items():
        _require(
            isinstance(name, str)
            and isinstance(record, dict)
            and set(record) == {"path", "sha256"}
            and isinstance(record["path"], str)
            and bool(record["path"]),
            f"Gate-A upstream authority record is invalid: {name}",
        )
        _hash(record["sha256"], f"upstream authority hash {name}")
    sources = value["sources"]
    outputs = value["outputs"]
    _require(
        isinstance(sources, dict) and bool(sources), "Gate-A source plan has no sources"
    )
    _require(
        isinstance(outputs, dict) and len(outputs) == 10,
        "Gate-A source plan must define exactly ten outputs",
    )
    required_outputs = {
        f"validation_{kind}_{part}"
        for kind in ("sessions", "market", "returns", "factors", "delistings")
        for part in ("main", "reserve")
    }
    _require(
        set(outputs) == required_outputs, "Gate-A normalized output role set differs"
    )
    for name, record in outputs.items():
        _require(isinstance(record, dict), f"Gate-A output record is invalid: {name}")
        part = "reserve" if name.endswith("_reserve") else "main"
        minimum, maximum = (
            (RESERVE_START.isoformat(), RESERVE_END.isoformat())
            if part == "reserve"
            else (MAIN_START.isoformat(), MAIN_END.isoformat())
        )
        _require(
            record.get("minimum_date") == minimum
            and record.get("maximum_date") == maximum,
            f"Gate-A output boundary differs: {name}",
        )
        columns = record.get("allowed_columns")
        _require(
            isinstance(columns, list) and columns and len(columns) == len(set(columns)),
            f"Gate-A output allowlist differs: {name}",
        )
        _hash(record.get("schema_sha256"), f"{name}.schema_sha256")
        _require(
            record.get("predictor_use_permitted") is False,
            f"Gate-A reserve/predictor boundary differs: {name}",
        )
    code = value["implementation_code_hashes"]
    _require(
        isinstance(code, dict) and bool(code), "Gate-A implementation hash set is empty"
    )
    for relative, digest in code.items():
        _require(
            isinstance(relative, str)
            and Path(relative).as_posix() == relative
            and not Path(relative).is_absolute()
            and ".." not in Path(relative).parts,
            "Gate-A implementation path is unsafe",
        )
        _hash(digest, f"implementation hash {relative}")
        path = PROJECT_ROOT / relative
        _require(
            path.is_file() and not path.is_symlink(),
            f"Gate-A implementation file is absent: {relative}",
        )
        _require(
            _implementation_sha256(relative, path) == digest,
            f"Gate-A implementation changed: {relative}",
        )
    payload = {key: value[key] for key in sorted(set(value) - {"payload_sha256"})}
    _require(
        value["payload_sha256"] == sha256_bytes(canonical_json_bytes(payload)),
        "Gate-A source plan payload digest differs",
    )
    return MappingProxyType(dict(value))


class _Lease:
    __slots__ = ("active",)

    def __init__(self) -> None:
        self.active = True

    def revoke(self, token: object) -> None:
        if token is not _ISSUANCE_TOKEN:
            raise Stage11SourceMaterializationSealedError(
                "Gate-A lease cannot be changed by a caller"
            )
        self.active = False


_ISSUANCE_TOKEN = object()


@dataclass(frozen=True, slots=True, init=False)
class Stage11SourceMaterializationAuthority:
    """Privately issued authority valid only inside one descriptor-bound scope."""

    approval_file_sha256: str
    approval_payload_sha256: str
    plan_sha256: str
    plan_payload_sha256: str
    approval_text_sha256: str
    _plan: Mapping[str, Any]
    _lease: _Lease

    def __init__(
        self,
        *,
        approval_file_sha256: str,
        approval_payload_sha256: str,
        plan_sha256: str,
        plan_payload_sha256: str,
        approval_text_sha256: str,
        plan: Mapping[str, Any],
        lease: _Lease,
        _issuance_token: object,
    ) -> None:
        if _issuance_token is not _ISSUANCE_TOKEN:
            raise Stage11SourceMaterializationSealedError(
                "Gate-A authority must be issued by the repository gate"
            )
        object.__setattr__(self, "approval_file_sha256", approval_file_sha256)
        object.__setattr__(self, "approval_payload_sha256", approval_payload_sha256)
        object.__setattr__(self, "plan_sha256", plan_sha256)
        object.__setattr__(self, "plan_payload_sha256", plan_payload_sha256)
        object.__setattr__(self, "approval_text_sha256", approval_text_sha256)
        object.__setattr__(self, "_plan", MappingProxyType(dict(plan)))
        object.__setattr__(self, "_lease", lease)

    @property
    def plan(self) -> Mapping[str, Any]:
        require_live_stage11_source_materialization_authority(self)
        return self._plan

    @property
    def authority_sha256(self) -> str:
        require_live_stage11_source_materialization_authority(self)
        return sha256_bytes(
            canonical_json_bytes(
                {
                    "approval_file_sha256": self.approval_file_sha256,
                    "approval_payload_sha256": self.approval_payload_sha256,
                    "approval_text_sha256": self.approval_text_sha256,
                    "plan_payload_sha256": self.plan_payload_sha256,
                    "plan_sha256": self.plan_sha256,
                    "scope": APPROVAL_SCOPE,
                }
            )
        )


def require_live_stage11_source_materialization_authority(
    authority: object,
) -> Stage11SourceMaterializationAuthority:
    if type(authority) is not Stage11SourceMaterializationAuthority:
        raise Stage11SourceMaterializationSealedError(
            "a live Gate-A source authority is required"
        )
    if not authority._lease.active:
        raise Stage11SourceMaterializationSealedError(
            "the Gate-A source authority lease is closed"
        )
    return authority


def _decode_json(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Stage11SourceMaterializationSealedError(f"invalid {label} JSON") from exc
    _require(isinstance(value, dict), f"{label} is not a JSON object")
    return value


@contextlib.contextmanager
def open_stage11_validation_source_materialization() -> (
    Iterator[Stage11SourceMaterializationAuthority]
):
    """Open the exact Gate-A authority, or fail before any empirical lookup."""

    # This must remain the first gate.  In the shipped sealed state, not even
    # the approval filename or the extraction plan is statted.
    if (
        FROZEN_SOURCE_MATERIALIZATION_APPROVAL_FILE_SHA256 is None
        or FROZEN_SOURCE_MATERIALIZATION_APPROVAL_PAYLOAD_SHA256 is None
    ):
        raise Stage11SourceMaterializationSealedError(
            "Stage 11 Gate-A source materialization is sealed pending explicit user approval"
        )
    approval_file_hash = _hash(
        FROZEN_SOURCE_MATERIALIZATION_APPROVAL_FILE_SHA256,
        "frozen Gate-A approval file hash",
    )
    approval_payload_hash = _hash(
        FROZEN_SOURCE_MATERIALIZATION_APPROVAL_PAYLOAD_SHA256,
        "frozen Gate-A approval payload hash",
    )
    approval_bytes = _stable_regular_bytes(
        APPROVAL_PATH,
        expected_sha256=approval_file_hash,
        label="Gate-A approval",
    )
    approval = validate_approval_mapping(
        _decode_json(approval_bytes, "Gate-A approval")
    )
    _require(
        approval["payload_sha256"] == approval_payload_hash,
        "frozen Gate-A approval logical digest differs",
    )
    plan_hash = _hash(
        approval["source_materialization_plan_sha256"], "Gate-A plan hash"
    )
    plan_bytes = _stable_regular_bytes(
        PLAN_PATH, expected_sha256=plan_hash, label="Gate-A source plan"
    )
    plan = _validate_plan_shape(_decode_json(plan_bytes, "Gate-A source plan"))
    upstream = plan["upstream_authorities"]
    stage10 = upstream.get("stage10_verified_pointer")
    methodology = upstream.get("methodology_source")
    _require(
        isinstance(stage10, dict)
        and stage10.get("sha256") == approval["stage10_verified_pointer_sha256"]
        and isinstance(methodology, dict)
        and methodology.get("sha256") == approval["methodology_source_sha256"],
        "Gate-A approval upstream bindings differ from the fixed plan",
    )
    for name, record in upstream.items():
        path = Path(record["path"])
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        _stable_regular_bytes(
            path,
            expected_sha256=record["sha256"],
            label=f"Gate-A upstream authority {name}",
        )
    lease = _Lease()
    authority = Stage11SourceMaterializationAuthority(
        approval_file_sha256=approval_file_hash,
        approval_payload_sha256=approval_payload_hash,
        plan_sha256=plan_hash,
        plan_payload_sha256=str(plan["payload_sha256"]),
        approval_text_sha256=str(approval["approval_text_sha256"]),
        plan=plan,
        lease=lease,
        _issuance_token=_ISSUANCE_TOKEN,
    )
    try:
        yield authority
        require_live_stage11_source_materialization_authority(authority)
    finally:
        lease.revoke(_ISSUANCE_TOKEN)


__all__ = [
    "APPROVAL_CONTRACT_ID",
    "APPROVAL_PATH",
    "APPROVAL_RELATIVE_PATH",
    "APPROVAL_SCOPE",
    "FIRST_PROHIBITED_DATE",
    "FROZEN_SOURCE_MATERIALIZATION_APPROVAL_FILE_SHA256",
    "FROZEN_SOURCE_MATERIALIZATION_APPROVAL_PAYLOAD_SHA256",
    "MAIN_END",
    "MAIN_START",
    "PLAN_ID",
    "PLAN_PATH",
    "PLAN_RELATIVE_PATH",
    "PROJECT_ROOT",
    "RESERVE_END",
    "RESERVE_START",
    "Stage11SourceMaterializationAuthority",
    "Stage11SourceMaterializationSealedError",
    "canonical_json_bytes",
    "open_stage11_validation_source_materialization",
    "require_live_stage11_source_materialization_authority",
    "sha256_bytes",
    "sha256_file",
    "validate_approval_mapping",
]
