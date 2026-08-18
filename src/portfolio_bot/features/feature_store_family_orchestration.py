"""Bounded, resumable cache orchestration for all seven Stage 6 families.

This module stops at authenticated cache batches and per-family completion
receipts.  It does not assemble the raw feature store, processed registries,
quality outputs, a primary manifest, labels, models, or portfolios.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import stat
import tomllib
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

import pyarrow.parquet as pq

from portfolio_bot.data.secure_publication import (
    AuthenticatedBundleGuard,
    AuthenticatedPointerGuard,
    SecurePublicationError,
    SecureStagingDirectory,
)
from portfolio_bot.features.feature_store_contract import DEFAULT_POLICY
from portfolio_bot.features.feature_store_inputs import FeatureStoreInputs
from portfolio_bot.features.feature_store_materialization import (
    FamilyBatchReceipt,
    FeatureMaterializationRow,
    FeatureStoreBuildIdentity,
    authenticate_materialized_family_batch,
    compute_feature_store_build_identity,
    family_feature_names,
    materialize_feature_family_batch,
    reauthenticate_feature_store_build_identity,
    validate_precomputed_feature_store_build_identity,
)
from portfolio_bot.features.materialize_aggregate_state import (
    materialize_aggregate_state,
)
from portfolio_bot.features.materialize_crsp import CRSP_FAMILIES
from portfolio_bot.features.materialize_crsp_routing import (
    CrspSignalRouter,
    spool_crsp_signal_router,
)
from portfolio_bot.features.materialize_crsp_streaming import (
    iter_unfinalized_crsp_rows,
)
from portfolio_bot.features.materialize_fundamentals import (
    materialize_fundamentals,
)
from portfolio_bot.features.materialize_optionmetrics import (
    materialize_optionmetrics,
)
from portfolio_bot.features.materialize_ownership_events import (
    materialize_ownership_events,
)
from portfolio_bot.features.phase_03_contract import canonical_json_bytes


_COMPLETION_DIRECTORY = "_completion"
_COMPLETION_FILE = "_SUCCESS.json"
_COMPLETION_STATUS = "CACHE_FAMILY_COMPLETE"
_COMPLETION_KEYS = frozenset(
    {
        "schema_version",
        "contract_id",
        "policy_id",
        "build_id",
        "content_fingerprint",
        "location_kind",
        "status",
        "family",
        "feature_count",
        "batch_count",
        "row_count",
        "lineage_row_count",
        "first_key",
        "last_key",
        "batches",
    }
)
_BATCH_KEYS = frozenset(
    {
        "ordinal",
        "batch_id",
        "row_count",
        "lineage_row_count",
        "first_key",
        "last_key",
        "success_sha256",
        "success_size_bytes",
    }
)
_KEY_KEYS = frozenset({"signal_date", "permno"})
_ALL_FAMILIES = (
    *CRSP_FAMILIES,
    "valuation_fundamentals",
    "ownership_filings_events_segments",
    "optionmetrics",
    "aggregate_state",
)

ProgressCallback = Callable[[Mapping[str, Any]], None]
SignalKey = tuple[dt.date, int]


class Stage6FamilyCacheOrchestrationError(RuntimeError):
    """Raised when cache orchestration cannot preserve the frozen spine."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Stage6FamilyCacheOrchestrationError(message)


@dataclass(frozen=True)
class FamilyBatchSummary:
    ordinal: int
    batch_id: str
    row_count: int
    lineage_row_count: int
    first_key: SignalKey
    last_key: SignalKey
    success_sha256: str
    success_size_bytes: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "ordinal": self.ordinal,
            "batch_id": self.batch_id,
            "row_count": self.row_count,
            "lineage_row_count": self.lineage_row_count,
            "first_key": _key_payload(self.first_key),
            "last_key": _key_payload(self.last_key),
            "success_sha256": self.success_sha256,
            "success_size_bytes": self.success_size_bytes,
        }


@dataclass(frozen=True)
class FamilyCompletionReceipt:
    path: Path
    family: str
    build_id: str
    feature_count: int
    batch_count: int
    row_count: int
    lineage_row_count: int
    first_key: SignalKey
    last_key: SignalKey
    batches: tuple[FamilyBatchSummary, ...]
    manifest_sha256: str
    manifest_size_bytes: int
    reused_existing: bool


@dataclass(frozen=True)
class Stage6FamilyCacheResult:
    build_identity: FeatureStoreBuildIdentity
    family_receipts: tuple[FamilyCompletionReceipt, ...]
    crsp_router: CrspSignalRouter | None
    final_identity_reauthenticated: bool

    @property
    def family_receipt_map(self) -> Mapping[str, FamilyCompletionReceipt]:
        return MappingProxyType(
            {item.family: item for item in self.family_receipts}
        )


def _load_policy(policy_path: Path) -> dict[str, Any]:
    with policy_path.open("rb") as handle:
        value = tomllib.load(handle)
    _require(isinstance(value, dict), "Feature-store policy is not an object")
    return value


def _family_order(policy_path: Path) -> tuple[str, ...]:
    policy = _load_policy(policy_path)
    order = tuple(str(value) for value in policy["build"]["family_order"])
    _require(
        order == _ALL_FAMILIES,
        "Frozen seven-family order differs",
    )
    return order


def _required_spine_rows(policy_path: Path) -> int:
    value = _load_policy(policy_path)["scope"]["required_spine_rows"]
    _require(
        isinstance(value, int)
        and not isinstance(value, bool)
        and value > 0,
        "Frozen required_spine_rows is invalid",
    )
    return value


def _cache_root(
    inputs: FeatureStoreInputs,
    policy_path: Path,
) -> Path:
    policy = _load_policy(policy_path)
    relative = Path(str(policy["database"]["cache_root"]))
    _require(
        not relative.is_absolute() and ".." not in relative.parts,
        "Frozen cache root is unsafe",
    )
    return Path(
        os.path.abspath(inputs.database_root.expanduser())
    ) / relative


def _family_root(
    inputs: FeatureStoreInputs,
    identity: FeatureStoreBuildIdentity,
    family: str,
    policy_path: Path,
) -> Path:
    return (
        _cache_root(inputs, policy_path)
        / "family_materialization"
        / identity.build_id
        / family
    )


def _completion_path(
    inputs: FeatureStoreInputs,
    identity: FeatureStoreBuildIdentity,
    family: str,
    policy_path: Path,
) -> Path:
    return (
        _family_root(inputs, identity, family, policy_path)
        / _COMPLETION_DIRECTORY
    )


def _emit(
    progress: ProgressCallback | None,
    event: str,
    **payload: Any,
) -> None:
    if progress is not None:
        progress(MappingProxyType({"event": event, **payload}))


def _key(row: FeatureMaterializationRow) -> SignalKey:
    _require(
        isinstance(row, FeatureMaterializationRow),
        "Family stream emitted an invalid row type",
    )
    _require(
        isinstance(row.signal_date, dt.date)
        and not isinstance(row.signal_date, dt.datetime)
        and isinstance(row.permno, int)
        and not isinstance(row.permno, bool),
        "Family stream emitted an invalid signal key",
    )
    return row.signal_date, row.permno


def _key_payload(value: SignalKey) -> dict[str, Any]:
    return {"signal_date": value[0].isoformat(), "permno": value[1]}


def _parsed_key(value: Any, label: str) -> SignalKey:
    _require(
        isinstance(value, dict) and set(value) == _KEY_KEYS,
        f"{label} key object differs",
    )
    try:
        signal = dt.date.fromisoformat(str(value["signal_date"]))
    except (TypeError, ValueError) as exc:
        raise Stage6FamilyCacheOrchestrationError(
            f"{label} signal_date is invalid"
        ) from exc
    permno = value["permno"]
    _require(
        isinstance(permno, int) and not isinstance(permno, bool),
        f"{label} permno is invalid",
    )
    return signal, int(permno)


def _batch_key_range(
    receipt: FamilyBatchReceipt,
    family: str,
) -> tuple[SignalKey, SignalKey, int]:
    value_path = receipt.path / f"feature_values_{family}.parquet"
    try:
        keys = pq.read_table(
            value_path,
            columns=["signal_date", "permno"],
            use_threads=False,
        )
    except Exception as exc:
        raise Stage6FamilyCacheOrchestrationError(
            f"Cannot inspect cache batch keys: {family}/{receipt.batch_id}"
        ) from exc
    values = list(
        zip(
            keys["signal_date"].to_pylist(),
            keys["permno"].to_pylist(),
            strict=True,
        )
    )
    _require(
        len(values) == receipt.row_count and bool(values),
        f"Cache batch key count differs: {family}/{receipt.batch_id}",
    )
    _require(
        all(
            isinstance(signal, dt.date)
            and isinstance(permno, int)
            and (
                index == 0
                or values[index - 1] < (signal, permno)
            )
            for index, (signal, permno) in enumerate(values)
        ),
        f"Cache batch keys are not strictly increasing: "
        f"{family}/{receipt.batch_id}",
    )
    return values[0], values[-1], len(values)


def _completion_payload(
    identity: FeatureStoreBuildIdentity,
    family: str,
    feature_count: int,
    batches: Sequence[FamilyBatchSummary],
) -> dict[str, Any]:
    _require(bool(batches), f"Completed family has no batches: {family}")
    row_count = sum(item.row_count for item in batches)
    lineage_count = sum(item.lineage_row_count for item in batches)
    return {
        "schema_version": 1,
        "contract_id": identity.contract_id,
        "policy_id": identity.policy_id,
        "build_id": identity.build_id,
        "content_fingerprint": identity.content_fingerprint,
        "location_kind": "cache_only_family_completion",
        "status": _COMPLETION_STATUS,
        "family": family,
        "feature_count": feature_count,
        "batch_count": len(batches),
        "row_count": row_count,
        "lineage_row_count": lineage_count,
        "first_key": _key_payload(batches[0].first_key),
        "last_key": _key_payload(batches[-1].last_key),
        "batches": [item.as_dict() for item in batches],
    }


def _parse_completion_payload(
    payload: bytes,
    *,
    identity: FeatureStoreBuildIdentity,
    family: str,
    expected_rows: int,
    policy_path: Path,
) -> tuple[dict[str, Any], tuple[FamilyBatchSummary, ...]]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Stage6FamilyCacheOrchestrationError(
            "Family completion manifest is not valid JSON"
        ) from exc
    _require(
        isinstance(value, dict) and set(value) == _COMPLETION_KEYS,
        "Family completion manifest keys differ",
    )
    expected = {
        "schema_version": 1,
        "contract_id": identity.contract_id,
        "policy_id": identity.policy_id,
        "build_id": identity.build_id,
        "content_fingerprint": identity.content_fingerprint,
        "location_kind": "cache_only_family_completion",
        "status": _COMPLETION_STATUS,
        "family": family,
        "feature_count": len(family_feature_names(family, policy_path)),
        "row_count": expected_rows,
    }
    for name, expected_value in expected.items():
        _require(
            value.get(name) == expected_value,
            f"Family completion binding differs: {name}",
        )
    raw_batches = value["batches"]
    _require(
        isinstance(raw_batches, list) and bool(raw_batches),
        "Family completion has no batches",
    )
    parsed: list[FamilyBatchSummary] = []
    for ordinal, raw in enumerate(raw_batches):
        _require(
            isinstance(raw, dict) and set(raw) == _BATCH_KEYS,
            "Family completion batch keys differ",
        )
        _require(
            raw["ordinal"] == ordinal
            and raw["batch_id"] == f"batch-{ordinal:06d}",
            "Family completion batch order differs",
        )
        for name in (
            "row_count",
            "lineage_row_count",
            "success_size_bytes",
        ):
            _require(
                isinstance(raw[name], int)
                and not isinstance(raw[name], bool)
                and raw[name] > 0,
                f"Family completion {name} is invalid",
            )
        digest = str(raw["success_sha256"])
        _require(
            len(digest) == 64
            and all(character in "0123456789abcdef" for character in digest),
            "Family completion batch hash is invalid",
        )
        parsed.append(
            FamilyBatchSummary(
                ordinal=ordinal,
                batch_id=str(raw["batch_id"]),
                row_count=int(raw["row_count"]),
                lineage_row_count=int(raw["lineage_row_count"]),
                first_key=_parsed_key(raw["first_key"], "batch first"),
                last_key=_parsed_key(raw["last_key"], "batch last"),
                success_sha256=digest,
                success_size_bytes=int(raw["success_size_bytes"]),
            )
        )
    feature_count = int(value["feature_count"])
    _require(
        value["batch_count"] == len(parsed)
        and value["lineage_row_count"] == expected_rows * feature_count
        and sum(item.row_count for item in parsed) == expected_rows
        and sum(item.lineage_row_count for item in parsed)
        == expected_rows * feature_count,
        "Family completion totals differ",
    )
    _require(
        value["first_key"] == _key_payload(parsed[0].first_key)
        and value["last_key"] == _key_payload(parsed[-1].last_key),
        "Family completion outer key range differs",
    )
    prior: SignalKey | None = None
    for item in parsed:
        _require(
            item.first_key <= item.last_key
            and (prior is None or prior < item.first_key)
            and item.lineage_row_count
            == item.row_count * feature_count,
            "Family completion batch key range or lineage count differs",
        )
        prior = item.last_key
    return value, tuple(parsed)


def _validate_family_children(
    root: Path,
    batches: Sequence[FamilyBatchSummary],
) -> None:
    try:
        observed = os.lstat(root)
        names = set(os.listdir(root))
    except OSError as exc:
        raise Stage6FamilyCacheOrchestrationError(
            "Cannot inspect completed family cache root"
        ) from exc
    _require(
        stat.S_ISDIR(observed.st_mode) and not stat.S_ISLNK(observed.st_mode),
        "Completed family cache root is not a real directory",
    )
    expected = {
        _COMPLETION_DIRECTORY,
        *(item.batch_id for item in batches),
    }
    _require(
        names == expected,
        "Completed family cache root has an open or extra child set",
    )
    for name in expected:
        selected = os.lstat(root / name)
        _require(
            stat.S_ISDIR(selected.st_mode)
            and not stat.S_ISLNK(selected.st_mode),
            "Completed family cache child is not a real directory",
        )


def authenticate_family_completion(
    *,
    inputs: FeatureStoreInputs,
    identity: FeatureStoreBuildIdentity,
    family: str,
    policy_path: Path = DEFAULT_POLICY,
) -> FamilyCompletionReceipt:
    """Authenticate one complete family's manifest and every bound batch."""

    validate_precomputed_feature_store_build_identity(
        inputs,
        identity,
        policy_path,
    )
    rows_required = _required_spine_rows(policy_path)
    _require(
        isinstance(rows_required, int)
        and not isinstance(rows_required, bool)
        and rows_required > 0,
        "Expected family row count is invalid",
    )
    cache_root = _cache_root(inputs, policy_path)
    completion = _completion_path(
        inputs,
        identity,
        family,
        policy_path,
    )
    manifest = completion / _COMPLETION_FILE
    try:
        with AuthenticatedPointerGuard.open(
            trusted_root=cache_root,
            path=manifest,
            label="family completion manifest",
        ) as pointer:
            _value, batches = _parse_completion_payload(
                pointer.payload,
                identity=identity,
                family=family,
                expected_rows=rows_required,
                policy_path=policy_path,
            )
            with AuthenticatedBundleGuard.open(
                trusted_root=cache_root,
                path=completion,
                expected_files=(_COMPLETION_FILE,),
                expected_hashes={
                    _COMPLETION_FILE: pointer.payload_sha256
                },
                label="family completion bundle",
                _permit_authenticated_directory_ctime_refresh=True,
            ) as bundle:
                pointer.gate()
                bundle.gate()
            manifest_sha256 = pointer.payload_sha256
            manifest_size = pointer.record.size_bytes
    except (
        SecurePublicationError,
        Stage6FamilyCacheOrchestrationError,
    ):
        raise
    except Exception as exc:
        raise Stage6FamilyCacheOrchestrationError(
            f"Cannot authenticate family completion: {family}"
        ) from exc

    _validate_family_children(
        _family_root(inputs, identity, family, policy_path),
        batches,
    )
    feature_count = len(family_feature_names(family, policy_path))
    for summary in batches:
        receipt = authenticate_materialized_family_batch(
            inputs=inputs,
            family=family,
            batch_id=summary.batch_id,
            policy_path=policy_path,
            build_identity=identity,
        )
        _require(
            receipt.row_count == summary.row_count
            and receipt.lineage_row_count == summary.lineage_row_count
            and receipt.success_sha256 == summary.success_sha256
            and receipt.success_size_bytes == summary.success_size_bytes,
            f"Family completion batch receipt differs: "
            f"{family}/{summary.batch_id}",
        )
        first_key, last_key, row_count = _batch_key_range(
            receipt,
            family,
        )
        _require(
            row_count == summary.row_count
            and first_key == summary.first_key
            and last_key == summary.last_key,
            f"Family completion batch key range differs: "
            f"{family}/{summary.batch_id}",
        )
    return FamilyCompletionReceipt(
        path=completion,
        family=family,
        build_id=identity.build_id,
        feature_count=feature_count,
        batch_count=len(batches),
        row_count=rows_required,
        lineage_row_count=rows_required * feature_count,
        first_key=batches[0].first_key,
        last_key=batches[-1].last_key,
        batches=batches,
        manifest_sha256=manifest_sha256,
        manifest_size_bytes=manifest_size,
        reused_existing=True,
    )


def _publish_family_completion(
    *,
    inputs: FeatureStoreInputs,
    identity: FeatureStoreBuildIdentity,
    family: str,
    batches: Sequence[FamilyBatchSummary],
    expected_rows: int,
    policy_path: Path,
) -> FamilyCompletionReceipt:
    feature_count = len(family_feature_names(family, policy_path))
    payload = _completion_payload(
        identity,
        family,
        feature_count,
        batches,
    )
    _require(
        payload["row_count"] == expected_rows
        and payload["lineage_row_count"]
        == expected_rows * feature_count,
        f"Cannot complete family with wrong totals: {family}",
    )
    bytes_value = canonical_json_bytes(payload)
    destination = _completion_path(
        inputs,
        identity,
        family,
        policy_path,
    )
    try:
        with SecureStagingDirectory.create(
            trusted_root=inputs.database_root,
            destination=destination,
            expected_files=(_COMPLETION_FILE,),
            label="Stage 6 family completion",
        ) as stage:
            with stage.open_exclusive_binary(_COMPLETION_FILE) as stream:
                stream.write(bytes_value)
            digest = hashlib.sha256(bytes_value).hexdigest()
            snapshot = stage.snapshot(
                expected_hashes={_COMPLETION_FILE: digest}
            )
            _require(
                snapshot.hashes[_COMPLETION_FILE] == digest,
                "Family completion staged hash differs",
            )

            def identity_gate() -> None:
                validate_precomputed_feature_store_build_identity(
                    inputs,
                    identity,
                    policy_path,
                )

            published = stage.publish(
                expected_hashes=snapshot.hashes,
                pre_publish_gate=identity_gate,
                post_publish_gate=identity_gate,
            )
    except (
        SecurePublicationError,
        Stage6FamilyCacheOrchestrationError,
    ):
        raise
    authenticated = authenticate_family_completion(
        inputs=inputs,
        identity=identity,
        family=family,
        policy_path=policy_path,
    )
    return FamilyCompletionReceipt(
        **{
            **authenticated.__dict__,
            "reused_existing": published.reused_existing,
        }
    )


class _FamilyAccumulator:
    def __init__(
        self,
        *,
        inputs: FeatureStoreInputs,
        identity: FeatureStoreBuildIdentity,
        family: str,
        expected_rows: int,
        policy_path: Path,
        progress: ProgressCallback | None,
    ) -> None:
        self.inputs = inputs
        self.identity = identity
        self.family = family
        self.expected_rows = expected_rows
        self.policy_path = policy_path
        self.progress = progress
        self.feature_count = len(
            family_feature_names(family, policy_path)
        )
        self.batches: list[FamilyBatchSummary] = []
        self.row_count = 0
        self.last_key: SignalKey | None = None
        _emit(progress, "family_started", family=family)

    def add(
        self,
        rows: Iterable[FeatureMaterializationRow],
    ) -> None:
        materialized = tuple(rows)
        _require(
            bool(materialized),
            f"Family source emitted an empty batch: {self.family}",
        )
        keys = tuple(_key(row) for row in materialized)
        _require(
            all(
                index == 0 or keys[index - 1] < key
                for index, key in enumerate(keys)
            ),
            f"Family source batch is not strictly key-sorted: {self.family}",
        )
        _require(
            self.last_key is None or self.last_key < keys[0],
            f"Family source key ranges are not monotone: {self.family}",
        )
        ordinal = len(self.batches)
        batch_id = f"batch-{ordinal:06d}"
        receipt = materialize_feature_family_batch(
            inputs=self.inputs,
            family=self.family,
            batch_id=batch_id,
            rows=materialized,
            policy_path=self.policy_path,
            build_identity=self.identity,
        )
        _require(
            receipt.row_count == len(materialized)
            and receipt.lineage_row_count
            == len(materialized) * self.feature_count,
            f"Common family batch totals differ: "
            f"{self.family}/{batch_id}",
        )
        summary = FamilyBatchSummary(
            ordinal=ordinal,
            batch_id=batch_id,
            row_count=receipt.row_count,
            lineage_row_count=receipt.lineage_row_count,
            first_key=keys[0],
            last_key=keys[-1],
            success_sha256=receipt.success_sha256,
            success_size_bytes=receipt.success_size_bytes,
        )
        self.batches.append(summary)
        self.row_count += len(materialized)
        self.last_key = keys[-1]
        _require(
            self.row_count <= self.expected_rows,
            f"Family source exceeded the frozen spine: {self.family}",
        )
        _emit(
            self.progress,
            "family_batch_complete",
            family=self.family,
            batch_id=batch_id,
            batch_rows=len(materialized),
            cumulative_rows=self.row_count,
            reused_existing=receipt.reused_existing,
            first_key=_key_payload(keys[0]),
            last_key=_key_payload(keys[-1]),
        )

    def finish(self) -> FamilyCompletionReceipt:
        _require(
            self.row_count == self.expected_rows,
            f"Family source emitted {self.row_count} rows; "
            f"expected {self.expected_rows}: {self.family}",
        )
        receipt = _publish_family_completion(
            inputs=self.inputs,
            identity=self.identity,
            family=self.family,
            batches=self.batches,
            expected_rows=self.expected_rows,
            policy_path=self.policy_path,
        )
        _emit(
            self.progress,
            "family_complete",
            family=self.family,
            batches=receipt.batch_count,
            rows=receipt.row_count,
            lineage_rows=receipt.lineage_row_count,
            reused_existing=receipt.reused_existing,
        )
        return receipt


def _completion_exists(
    *,
    inputs: FeatureStoreInputs,
    identity: FeatureStoreBuildIdentity,
    family: str,
    policy_path: Path,
) -> bool:
    return os.path.lexists(
        _completion_path(
            inputs,
            identity,
            family,
            policy_path,
        )
    )


def _load_existing_completions(
    *,
    inputs: FeatureStoreInputs,
    identity: FeatureStoreBuildIdentity,
    families: Sequence[str],
    expected_rows: int,
    policy_path: Path,
    progress: ProgressCallback | None,
) -> dict[str, FamilyCompletionReceipt]:
    result: dict[str, FamilyCompletionReceipt] = {}
    for family in families:
        if not _completion_exists(
            inputs=inputs,
            identity=identity,
            family=family,
            policy_path=policy_path,
        ):
            continue
        receipt = authenticate_family_completion(
            inputs=inputs,
            identity=identity,
            family=family,
            policy_path=policy_path,
        )
        result[family] = receipt
        _emit(
            progress,
            "family_resumed_complete",
            family=family,
            batches=receipt.batch_count,
            rows=receipt.row_count,
            lineage_rows=receipt.lineage_row_count,
        )
    return result


def _validate_spool_path(
    supplied: Path | None,
    *,
    cache_root: Path,
    build_id: str,
) -> Path:
    candidate = (
        cache_root / "crsp_signal_router" / build_id
        if supplied is None
        else Path(os.path.abspath(Path(supplied).expanduser()))
    )
    root = Path(os.path.abspath(cache_root.expanduser()))
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise Stage6FamilyCacheOrchestrationError(
            "CRSP spool path must remain beneath the Stage 6 cache root"
        ) from exc
    return candidate


def _materialize_crsp_families(
    *,
    inputs: FeatureStoreInputs,
    identity: FeatureStoreBuildIdentity,
    pending: Sequence[str],
    existing: Mapping[str, FamilyCompletionReceipt],
    expected_rows: int,
    batch_size: int,
    checkpoint_rows: int,
    spool_path: Path,
    policy_path: Path,
    progress: ProgressCallback | None,
) -> tuple[dict[str, FamilyCompletionReceipt], CrspSignalRouter | None]:
    result = dict(existing)
    if not pending:
        return result, None
    _emit(
        progress,
        "crsp_router_started",
        spool_path=spool_path.as_posix(),
    )

    def source(
        checkpoint: tuple[int, dt.date] | None,
    ):
        return iter_unfinalized_crsp_rows(
            inputs,
            start_after_key=checkpoint,
            progress=lambda value: _emit(
                progress,
                "crsp_scan_progress",
                **dict(value),
            ),
        )

    router = spool_crsp_signal_router(
        source,
        spool_path=spool_path,
        source_id=identity.content_fingerprint,
        expected_rows=expected_rows,
        checkpoint_rows=checkpoint_rows,
    )
    _emit(
        progress,
        "crsp_router_complete",
        rows=router.row_count,
        signal_dates=len(router.signal_dates),
        reused_existing=router.reused_existing,
    )
    accumulators = {
        family: _FamilyAccumulator(
            inputs=inputs,
            identity=identity,
            family=family,
            expected_rows=expected_rows,
            policy_path=policy_path,
            progress=progress,
        )
        for family in pending
    }
    buffers: dict[str, list[FeatureMaterializationRow]] = {
        family: [] for family in pending
    }
    for routed in router.iter_family_row_batches(
        batch_size=batch_size
    ):
        accumulator = accumulators.get(routed.family)
        if accumulator is not None:
            buffer = buffers[routed.family]
            buffer.extend(routed.rows)
            while len(buffer) >= batch_size:
                accumulator.add(buffer[:batch_size])
                del buffer[:batch_size]
    for family in CRSP_FAMILIES:
        accumulator = accumulators.get(family)
        if accumulator is not None:
            if buffers[family]:
                accumulator.add(buffers[family])
            result[family] = accumulator.finish()
    return result, router


def _materialize_stream_family(
    *,
    inputs: FeatureStoreInputs,
    identity: FeatureStoreBuildIdentity,
    family: str,
    batches: Iterable[Iterable[FeatureMaterializationRow]],
    expected_rows: int,
    policy_path: Path,
    progress: ProgressCallback | None,
) -> FamilyCompletionReceipt:
    accumulator = _FamilyAccumulator(
        inputs=inputs,
        identity=identity,
        family=family,
        expected_rows=expected_rows,
        policy_path=policy_path,
        progress=progress,
    )
    for batch in batches:
        accumulator.add(batch)
    return accumulator.finish()


def materialize_stage6_family_cache(
    inputs: FeatureStoreInputs,
    *,
    batch_size: int = 20_000,
    crsp_checkpoint_rows: int = 4_096,
    crsp_spool_path: Path | None = None,
    policy_path: Path = DEFAULT_POLICY,
    progress: ProgressCallback | None = None,
) -> Stage6FamilyCacheResult:
    """Materialize/resume all seven cache families and stop before assembly."""

    _require(
        isinstance(inputs, FeatureStoreInputs),
        "Stage 6 orchestration requires FeatureStoreInputs",
    )
    for value, label in (
        (batch_size, "batch_size"),
        (crsp_checkpoint_rows, "crsp_checkpoint_rows"),
    ):
        _require(
            isinstance(value, int)
            and not isinstance(value, bool)
            and value > 0,
            f"{label} must be a positive integer",
        )
    family_order = _family_order(policy_path)
    expected_rows = _required_spine_rows(policy_path)
    identity = compute_feature_store_build_identity(inputs, policy_path)
    _emit(
        progress,
        "identity_authenticated",
        build_id=identity.build_id,
        inputs=len(identity.input_receipts),
    )
    existing = _load_existing_completions(
        inputs=inputs,
        identity=identity,
        families=family_order,
        expected_rows=expected_rows,
        policy_path=policy_path,
        progress=progress,
    )
    crsp_pending = [
        family for family in CRSP_FAMILIES if family not in existing
    ]
    cache_root = _cache_root(inputs, policy_path)
    spool = _validate_spool_path(
        crsp_spool_path,
        cache_root=cache_root,
        build_id=identity.build_id,
    )
    receipts, router = _materialize_crsp_families(
        inputs=inputs,
        identity=identity,
        pending=crsp_pending,
        existing={
            family: existing[family]
            for family in CRSP_FAMILIES
            if family in existing
        },
        expected_rows=expected_rows,
        batch_size=batch_size,
        checkpoint_rows=crsp_checkpoint_rows,
        spool_path=spool,
        policy_path=policy_path,
        progress=progress,
    )
    for family in family_order[3:]:
        if family in existing:
            receipts[family] = existing[family]
            continue
        if family == "valuation_fundamentals":
            batches = materialize_fundamentals(
                inputs,
                batch_size=batch_size,
                policy_path=policy_path,
                build_identity=identity,
            )
        elif family == "ownership_filings_events_segments":
            batches = materialize_ownership_events(
                inputs,
                batch_size=batch_size,
                policy_path=policy_path,
                build_identity=identity,
            )
        elif family == "optionmetrics":
            batches = materialize_optionmetrics(
                inputs,
                batch_size=batch_size,
                policy_path=policy_path,
                build_identity=identity,
            )
        elif family == "aggregate_state":
            batches = materialize_aggregate_state(
                inputs,
                batch_size=batch_size,
            ).iter_materialization_row_batches()
        else:
            raise Stage6FamilyCacheOrchestrationError(
                f"Unknown Stage 6 family source: {family}"
            )
        receipts[family] = _materialize_stream_family(
            inputs=inputs,
            identity=identity,
            family=family,
            batches=batches,
            expected_rows=expected_rows,
            policy_path=policy_path,
            progress=progress,
        )
    _require(
        set(receipts) == set(family_order),
        "Stage 6 cache orchestration did not complete seven families",
    )
    reauthenticate_feature_store_build_identity(
        inputs,
        identity,
        policy_path,
    )
    _emit(
        progress,
        "identity_reauthenticated",
        build_id=identity.build_id,
    )
    ordered = tuple(receipts[family] for family in family_order)
    _emit(
        progress,
        "orchestration_complete",
        build_id=identity.build_id,
        families=len(ordered),
        rows_per_family=expected_rows,
        lineage_rows=sum(item.lineage_row_count for item in ordered),
    )
    return Stage6FamilyCacheResult(
        build_identity=identity,
        family_receipts=ordered,
        crsp_router=router,
        final_identity_reauthenticated=True,
    )
