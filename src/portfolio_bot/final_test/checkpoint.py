"""Content-addressed monthly checkpoints for resumable Stage 13 execution.

Each checkpoint binds its exact execution plan, month, state bytes, and prior
checkpoint digest.  A per-plan atomic pointer selects the latest compatible
chain head.  Incomplete orphan objects are harmless: resume follows only the
last authenticated pointer and never restarts an already committed month.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from portfolio_bot.final_test.authorization import Stage12VerifiedBinding
from portfolio_bot.final_test.calendar import (
    FinalCalendarError,
    final_holding_month_descriptor,
    validate_final_month_sequence,
)


CHECKPOINT_SCHEMA_VERSION = 1
LATEST_POINTER_SCHEMA_VERSION = 1

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.:-]{2,127}$")
_STATE_FORMAT = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_.+/-]{0,127}$")
_RECORD_FIELDS = frozenset(
    {
        "schema_version",
        "chain_id",
        "plan_sha256",
        "holding_month",
        "ordinal",
        "previous_checkpoint_sha256",
        "state_sha256",
        "state_size_bytes",
        "state_format",
        "checkpoint_sha256",
    }
)
_POINTER_FIELDS = frozenset(
    {
        "schema_version",
        "plan_sha256",
        "checkpoint_sha256",
        "holding_month",
        "ordinal",
        "payload_sha256",
    }
)


class CheckpointError(RuntimeError):
    """Raised when a checkpoint chain is incompatible, incomplete, or altered."""


def _canonical_json_bytes(value: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_hash(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise CheckpointError(
            f"{field_name} must be 64 lowercase hexadecimal characters"
        )
    return value


def _require_identifier(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise CheckpointError(f"{field_name} is not a bounded identifier")
    return value


def _strict_fields(
    value: Mapping[str, object],
    expected: frozenset[str],
    *,
    object_name: str,
) -> None:
    fields = frozenset(value)
    if fields != expected:
        missing = sorted(expected - fields)
        extra = sorted(fields - expected)
        raise CheckpointError(
            f"{object_name} field mismatch; missing={missing}, extra={extra}"
        )


@dataclass(frozen=True, slots=True)
class CheckpointPlan:
    """All identities that make persisted monthly state reusable."""

    chain_id: str
    stage12_binding: Stage12VerifiedBinding
    execution_config_sha256: str
    holding_months: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_identifier(self.chain_id, field_name="chain_id")
        if not isinstance(self.stage12_binding, Stage12VerifiedBinding):
            raise CheckpointError(
                "stage12_binding must be a verified Stage 12 binding"
            )
        _require_hash(
            self.execution_config_sha256,
            field_name="execution_config_sha256",
        )
        try:
            months = validate_final_month_sequence(self.holding_months)
        except FinalCalendarError as exc:
            raise CheckpointError(str(exc)) from exc
        object.__setattr__(self, "holding_months", months)

    def to_mapping(self) -> dict[str, object]:
        return {
            "chain_id": self.chain_id,
            **self.stage12_binding.to_mapping(),
            "stage12_binding_sha256": self.stage12_binding.binding_sha256,
            "execution_config_sha256": self.execution_config_sha256,
            "holding_months": list(self.holding_months),
        }

    @property
    def plan_sha256(self) -> str:
        return _sha256_bytes(_canonical_json_bytes(self.to_mapping()))


def _record_payload(
    *,
    chain_id: str,
    plan_sha256: str,
    holding_month: str,
    ordinal: int,
    previous_checkpoint_sha256: str | None,
    state_sha256: str,
    state_size_bytes: int,
    state_format: str,
) -> dict[str, object]:
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "chain_id": chain_id,
        "plan_sha256": plan_sha256,
        "holding_month": holding_month,
        "ordinal": ordinal,
        "previous_checkpoint_sha256": previous_checkpoint_sha256,
        "state_sha256": state_sha256,
        "state_size_bytes": state_size_bytes,
        "state_format": state_format,
    }


@dataclass(frozen=True, slots=True)
class MonthlyCheckpointRecord:
    """Authenticated receipt for one committed month."""

    schema_version: int
    chain_id: str
    plan_sha256: str
    holding_month: str
    ordinal: int
    previous_checkpoint_sha256: str | None
    state_sha256: str
    state_size_bytes: int
    state_format: str
    checkpoint_sha256: str

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != CHECKPOINT_SCHEMA_VERSION
        ):
            raise CheckpointError("unsupported checkpoint schema version")
        _require_identifier(self.chain_id, field_name="chain_id")
        _require_hash(self.plan_sha256, field_name="plan_sha256")
        try:
            final_holding_month_descriptor(self.holding_month)
        except FinalCalendarError as exc:
            raise CheckpointError(str(exc)) from exc
        if type(self.ordinal) is not int or self.ordinal < 1:
            raise CheckpointError("ordinal must be a positive integer")
        if self.previous_checkpoint_sha256 is not None:
            _require_hash(
                self.previous_checkpoint_sha256,
                field_name="previous_checkpoint_sha256",
            )
        _require_hash(self.state_sha256, field_name="state_sha256")
        if type(self.state_size_bytes) is not int or self.state_size_bytes <= 0:
            raise CheckpointError("state_size_bytes must be a positive integer")
        if (
            not isinstance(self.state_format, str)
            or _STATE_FORMAT.fullmatch(self.state_format) is None
        ):
            raise CheckpointError("state_format is not a bounded format identifier")
        _require_hash(self.checkpoint_sha256, field_name="checkpoint_sha256")
        if self.checkpoint_sha256 != self.compute_checkpoint_sha256():
            raise CheckpointError("checkpoint content digest mismatch")

    def payload_without_digest(self) -> dict[str, object]:
        return _record_payload(
            chain_id=self.chain_id,
            plan_sha256=self.plan_sha256,
            holding_month=self.holding_month,
            ordinal=self.ordinal,
            previous_checkpoint_sha256=self.previous_checkpoint_sha256,
            state_sha256=self.state_sha256,
            state_size_bytes=self.state_size_bytes,
            state_format=self.state_format,
        )

    def compute_checkpoint_sha256(self) -> str:
        return _sha256_bytes(_canonical_json_bytes(self.payload_without_digest()))

    def to_mapping(self) -> dict[str, object]:
        value = self.payload_without_digest()
        value["checkpoint_sha256"] = self.checkpoint_sha256
        return value

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, object],
    ) -> "MonthlyCheckpointRecord":
        if not isinstance(value, Mapping):
            raise CheckpointError("checkpoint receipt must be a mapping")
        _strict_fields(value, _RECORD_FIELDS, object_name="checkpoint receipt")
        return cls(
            schema_version=value["schema_version"],  # type: ignore[arg-type]
            chain_id=value["chain_id"],  # type: ignore[arg-type]
            plan_sha256=value["plan_sha256"],  # type: ignore[arg-type]
            holding_month=value["holding_month"],  # type: ignore[arg-type]
            ordinal=value["ordinal"],  # type: ignore[arg-type]
            previous_checkpoint_sha256=value[
                "previous_checkpoint_sha256"
            ],  # type: ignore[arg-type]
            state_sha256=value["state_sha256"],  # type: ignore[arg-type]
            state_size_bytes=value["state_size_bytes"],  # type: ignore[arg-type]
            state_format=value["state_format"],  # type: ignore[arg-type]
            checkpoint_sha256=value["checkpoint_sha256"],  # type: ignore[arg-type]
        )


def _pointer_payload(
    *,
    plan_sha256: str,
    checkpoint_sha256: str,
    holding_month: str,
    ordinal: int,
) -> dict[str, object]:
    return {
        "schema_version": LATEST_POINTER_SCHEMA_VERSION,
        "plan_sha256": plan_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "holding_month": holding_month,
        "ordinal": ordinal,
    }


@dataclass(frozen=True, slots=True)
class LatestCheckpointPointer:
    schema_version: int
    plan_sha256: str
    checkpoint_sha256: str
    holding_month: str
    ordinal: int
    payload_sha256: str

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != LATEST_POINTER_SCHEMA_VERSION
        ):
            raise CheckpointError("unsupported latest-pointer schema version")
        _require_hash(self.plan_sha256, field_name="plan_sha256")
        _require_hash(
            self.checkpoint_sha256,
            field_name="checkpoint_sha256",
        )
        try:
            final_holding_month_descriptor(self.holding_month)
        except FinalCalendarError as exc:
            raise CheckpointError(str(exc)) from exc
        if type(self.ordinal) is not int or self.ordinal < 1:
            raise CheckpointError("pointer ordinal must be positive")
        _require_hash(self.payload_sha256, field_name="payload_sha256")
        if self.payload_sha256 != self.compute_payload_sha256():
            raise CheckpointError("latest-pointer payload digest mismatch")

    def payload_without_digest(self) -> dict[str, object]:
        return _pointer_payload(
            plan_sha256=self.plan_sha256,
            checkpoint_sha256=self.checkpoint_sha256,
            holding_month=self.holding_month,
            ordinal=self.ordinal,
        )

    def compute_payload_sha256(self) -> str:
        return _sha256_bytes(_canonical_json_bytes(self.payload_without_digest()))

    def to_mapping(self) -> dict[str, object]:
        value = self.payload_without_digest()
        value["payload_sha256"] = self.payload_sha256
        return value

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, object],
    ) -> "LatestCheckpointPointer":
        if not isinstance(value, Mapping):
            raise CheckpointError("latest checkpoint pointer must be a mapping")
        _strict_fields(value, _POINTER_FIELDS, object_name="latest pointer")
        return cls(
            schema_version=value["schema_version"],  # type: ignore[arg-type]
            plan_sha256=value["plan_sha256"],  # type: ignore[arg-type]
            checkpoint_sha256=value["checkpoint_sha256"],  # type: ignore[arg-type]
            holding_month=value["holding_month"],  # type: ignore[arg-type]
            ordinal=value["ordinal"],  # type: ignore[arg-type]
            payload_sha256=value["payload_sha256"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class LoadedCheckpoint:
    """Latest authenticated record and its exact persisted state bytes."""

    record: MonthlyCheckpointRecord
    state: bytes


def _load_json_mapping(path: Path, *, object_name: str) -> Mapping[str, object]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise CheckpointError(f"{object_name} could not be read") from exc
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckpointError(f"{object_name} is not valid JSON") from exc
    if not isinstance(value, Mapping):
        raise CheckpointError(f"{object_name} must contain a JSON object")
    return value


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


class MonthlyCheckpointStore:
    """Durable store that resumes only the exact compatible checkpoint chain."""

    def __init__(self, root: Path, plan: CheckpointPlan) -> None:
        if not isinstance(root, Path) or not root.is_absolute() or ".." in root.parts:
            raise CheckpointError("checkpoint root must be an absolute bounded path")
        if not isinstance(plan, CheckpointPlan):
            raise CheckpointError("plan must be a CheckpointPlan")
        self.root = root
        self.plan = plan

    @property
    def _objects_root(self) -> Path:
        return self.root / "objects"

    @property
    def _pointer_path(self) -> Path:
        return self.root / "chains" / self.plan.plan_sha256 / "LATEST.json"

    def _object_directory(self, checkpoint_sha256: str) -> Path:
        _require_hash(checkpoint_sha256, field_name="checkpoint_sha256")
        return self._objects_root / checkpoint_sha256

    def _load_record(
        self,
        checkpoint_sha256: str,
        *,
        load_state: bool,
    ) -> LoadedCheckpoint | MonthlyCheckpointRecord:
        object_directory = self._object_directory(checkpoint_sha256)
        receipt = MonthlyCheckpointRecord.from_mapping(
            _load_json_mapping(
                object_directory / "checkpoint.json",
                object_name="checkpoint receipt",
            )
        )
        if receipt.checkpoint_sha256 != checkpoint_sha256:
            raise CheckpointError(
                "checkpoint directory name differs from receipt digest"
            )
        if receipt.plan_sha256 != self.plan.plan_sha256:
            raise CheckpointError("checkpoint belongs to an incompatible plan")
        if receipt.chain_id != self.plan.chain_id:
            raise CheckpointError("checkpoint chain identifier differs")
        if receipt.ordinal > len(self.plan.holding_months):
            raise CheckpointError("checkpoint ordinal exceeds the planned calendar")
        expected_month = self.plan.holding_months[receipt.ordinal - 1]
        if receipt.holding_month != expected_month:
            raise CheckpointError("checkpoint month differs from planned ordinal")
        if not load_state:
            return receipt
        try:
            state = (object_directory / "state.bin").read_bytes()
        except OSError as exc:
            raise CheckpointError("checkpoint state could not be read") from exc
        if (
            len(state) != receipt.state_size_bytes
            or _sha256_bytes(state) != receipt.state_sha256
        ):
            raise CheckpointError("checkpoint state identity mismatch")
        return LoadedCheckpoint(record=receipt, state=state)

    def _validate_ancestry(self, head: MonthlyCheckpointRecord) -> None:
        current = head
        seen: set[str] = set()
        while True:
            if current.checkpoint_sha256 in seen:
                raise CheckpointError("checkpoint ancestry contains a cycle")
            seen.add(current.checkpoint_sha256)
            if current.ordinal == 1:
                if current.previous_checkpoint_sha256 is not None:
                    raise CheckpointError(
                        "genesis checkpoint cannot have a predecessor"
                    )
                return
            if current.previous_checkpoint_sha256 is None:
                raise CheckpointError(
                    "non-genesis checkpoint is missing its predecessor"
                )
            loaded_previous = self._load_record(
                current.previous_checkpoint_sha256,
                load_state=True,
            )
            if not isinstance(loaded_previous, LoadedCheckpoint):
                raise RuntimeError("internal checkpoint loader type mismatch")
            previous = loaded_previous.record
            if previous.ordinal != current.ordinal - 1:
                raise CheckpointError("checkpoint ancestry skips a planned month")
            current = previous

    def resume_from_latest(self) -> LoadedCheckpoint | None:
        """Return the last exact compatible state, or ``None`` before month one."""

        if not self._pointer_path.exists():
            return None
        pointer = LatestCheckpointPointer.from_mapping(
            _load_json_mapping(
                self._pointer_path,
                object_name="latest checkpoint pointer",
            )
        )
        if pointer.plan_sha256 != self.plan.plan_sha256:
            raise CheckpointError("latest pointer belongs to an incompatible plan")
        loaded = self._load_record(pointer.checkpoint_sha256, load_state=True)
        if not isinstance(loaded, LoadedCheckpoint):
            raise RuntimeError("internal checkpoint loader type mismatch")
        if (
            loaded.record.ordinal != pointer.ordinal
            or loaded.record.holding_month != pointer.holding_month
        ):
            raise CheckpointError("latest pointer and checkpoint head differ")
        self._validate_ancestry(loaded.record)
        return loaded

    def _persist_object(
        self,
        record: MonthlyCheckpointRecord,
        state: bytes,
    ) -> None:
        destination = self._object_directory(record.checkpoint_sha256)
        if destination.exists():
            loaded = self._load_record(record.checkpoint_sha256, load_state=True)
            if (
                not isinstance(loaded, LoadedCheckpoint)
                or loaded.record != record
                or loaded.state != state
            ):
                raise CheckpointError(
                    "existing content-addressed checkpoint differs"
                )
            return
        self._objects_root.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(prefix=".checkpoint.", dir=self._objects_root)
        )
        try:
            (temporary / "checkpoint.json").write_bytes(
                _canonical_json_bytes(record.to_mapping())
            )
            (temporary / "state.bin").write_bytes(state)
            try:
                os.replace(temporary, destination)
            except OSError:
                if not destination.exists():
                    raise
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        loaded = self._load_record(record.checkpoint_sha256, load_state=True)
        if (
            not isinstance(loaded, LoadedCheckpoint)
            or loaded.record != record
            or loaded.state != state
        ):
            raise CheckpointError("persisted checkpoint failed authentication")

    def commit(
        self,
        *,
        holding_month: str,
        state: bytes,
        state_format: str = "application/octet-stream",
    ) -> LoadedCheckpoint:
        """Commit exactly the next planned month and atomically advance resume."""

        if not isinstance(state, bytes) or not state:
            raise CheckpointError("checkpoint state must be nonempty bytes")
        if (
            not isinstance(state_format, str)
            or _STATE_FORMAT.fullmatch(state_format) is None
        ):
            raise CheckpointError("state_format is not a bounded format identifier")
        latest = self.resume_from_latest()
        state_sha256 = _sha256_bytes(state)
        if latest is not None and holding_month == latest.record.holding_month:
            if (
                latest.record.state_sha256 == state_sha256
                and latest.record.state_size_bytes == len(state)
                and latest.record.state_format == state_format
            ):
                return latest
            raise CheckpointError(
                "an already committed month cannot be replaced with new state"
            )
        next_ordinal = 1 if latest is None else latest.record.ordinal + 1
        if next_ordinal > len(self.plan.holding_months):
            raise CheckpointError("checkpoint plan is already complete")
        expected_month = self.plan.holding_months[next_ordinal - 1]
        if holding_month != expected_month:
            raise CheckpointError(
                f"next checkpoint must be {expected_month}; cannot skip or restart"
            )
        previous_sha256 = (
            None if latest is None else latest.record.checkpoint_sha256
        )
        payload = _record_payload(
            chain_id=self.plan.chain_id,
            plan_sha256=self.plan.plan_sha256,
            holding_month=holding_month,
            ordinal=next_ordinal,
            previous_checkpoint_sha256=previous_sha256,
            state_sha256=state_sha256,
            state_size_bytes=len(state),
            state_format=state_format,
        )
        record = MonthlyCheckpointRecord(
            schema_version=CHECKPOINT_SCHEMA_VERSION,
            chain_id=self.plan.chain_id,
            plan_sha256=self.plan.plan_sha256,
            holding_month=holding_month,
            ordinal=next_ordinal,
            previous_checkpoint_sha256=previous_sha256,
            state_sha256=state_sha256,
            state_size_bytes=len(state),
            state_format=state_format,
            checkpoint_sha256=_sha256_bytes(_canonical_json_bytes(payload)),
        )
        self._persist_object(record, state)
        pointer_payload = _pointer_payload(
            plan_sha256=self.plan.plan_sha256,
            checkpoint_sha256=record.checkpoint_sha256,
            holding_month=record.holding_month,
            ordinal=record.ordinal,
        )
        pointer = LatestCheckpointPointer(
            schema_version=LATEST_POINTER_SCHEMA_VERSION,
            plan_sha256=self.plan.plan_sha256,
            checkpoint_sha256=record.checkpoint_sha256,
            holding_month=record.holding_month,
            ordinal=record.ordinal,
            payload_sha256=_sha256_bytes(
                _canonical_json_bytes(pointer_payload)
            ),
        )
        _atomic_write(
            self._pointer_path,
            _canonical_json_bytes(pointer.to_mapping()),
        )
        loaded = self.resume_from_latest()
        if loaded is None or loaded.record != record or loaded.state != state:
            raise CheckpointError("latest checkpoint pointer did not authenticate")
        return loaded


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "CheckpointError",
    "CheckpointPlan",
    "LatestCheckpointPointer",
    "LoadedCheckpoint",
    "MonthlyCheckpointRecord",
    "MonthlyCheckpointStore",
]
