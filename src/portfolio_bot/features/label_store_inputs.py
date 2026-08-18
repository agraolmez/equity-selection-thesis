"""Resolve and authenticate the exact six-table Stage 7 input allowlist."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from portfolio_bot.features.label_store_contract import (
    DEFAULT_POLICY,
    LabelStoreContractError,
    PROJECT_ROOT,
    validate_label_store_contract,
)


@dataclass(frozen=True)
class LabelInputTable:
    gate: str
    name: str
    _path: Path
    sha256: str
    schema_sha256: str
    size_bytes: int
    rows: int
    row_groups: int
    projected_columns: tuple[str, ...]
    stat_identity: tuple[int, ...]


@dataclass(frozen=True)
class LabelStoreInputs:
    database_root: Path
    tables: Mapping[str, LabelInputTable]
    hashes_authenticated: bool

    def path(self, name: str) -> Path:
        self._require_authenticated()
        if name in {
            "rebalance_calendar.parquet",
            "trading_calendar.parquet",
            "crsp_security_day_return.parquet",
            "ff_factor_daily_observation.parquet",
        }:
            raise LabelStoreContractError(
                f"Temporal Stage 7 input requires bounded delivery: {name}"
            )
        return self._trusted_path(name)

    def _require_authenticated(self) -> None:
        if not self.hashes_authenticated:
            raise LabelStoreContractError(
                "Unauthenticated Stage 7 catalog cannot deliver data"
            )

    def _trusted_path(self, name: str) -> Path:
        self._require_authenticated()
        try:
            table = self.tables[name]
        except KeyError as exc:
            raise LabelStoreContractError(
                f"Unregistered Stage 7 input table: {name}"
            ) from exc
        path = _anchored_regular_path(
            table._path,
            trusted_root=self.database_root,
            label=f"held Stage 7 input {name}",
        )
        if _stat_identity(path.stat()) != table.stat_identity:
            raise LabelStoreContractError(
                f"Stage 7 input identity changed: {name}"
            )
        return path

    def read_development_calendar(self) -> pa.Table:
        """Return only the 178 authorized development holding intervals."""

        self._require_authenticated()
        table = self.tables["rebalance_calendar.parquet"]
        observed = pq.read_table(
            self._trusted_path(table.name),
            columns=list(table.projected_columns),
        )
        start = pa.scalar(dt.date(2000, 1, 31), type=pa.date32())
        last_signal = pa.scalar(
            dt.date(2014, 10, 31), type=pa.date32()
        )
        last_exit = pa.scalar(
            dt.date(2014, 12, 1), type=pa.date32()
        )
        mask = pc.and_(
            pc.equal(observed["stage"], "development"),
            pc.and_(
                pc.greater_equal(observed["signal_date"], start),
                pc.and_(
                    pc.less_equal(
                        observed["signal_date"], last_signal
                    ),
                    pc.less_equal(observed["exit_date"], last_exit),
                ),
            ),
        )
        bounded = observed.filter(pc.fill_null(mask, False))
        if (
            bounded.num_rows != 178
            or pc.any(bounded["is_embargo"]).as_py()
            or pc.any(bounded["is_partial"]).as_py()
        ):
            raise LabelStoreContractError(
                "Bounded Stage 7 rebalance calendar differs"
            )
        return bounded

    def read_development_trading_calendar(self) -> pa.Table:
        """Return verified sessions from 2000-01-03 through 2014-12-01."""

        self._require_authenticated()
        table = self.tables["trading_calendar.parquet"]
        observed = pq.read_table(
            self._trusted_path(table.name),
            columns=list(table.projected_columns),
        )
        start = pa.scalar(dt.date(2000, 1, 3), type=pa.date32())
        cutoff = pa.scalar(dt.date(2014, 12, 1), type=pa.date32())
        mask = pc.and_(
            pc.greater_equal(observed["calendar_date"], start),
            pc.less_equal(observed["calendar_date"], cutoff),
        )
        bounded = observed.filter(pc.fill_null(mask, False))
        if bounded.num_rows != 3_752:
            raise LabelStoreContractError(
                "Bounded Stage 7 trading calendar differs"
            )
        return bounded

    def read_development_factors(self) -> pa.Table:
        """Return the already-bounded daily FF5, UMD, and RF table."""

        self._require_authenticated()
        table = self.tables["ff_factor_daily_observation.parquet"]
        observed = pq.read_table(
            self._trusted_path(table.name),
            columns=list(table.projected_columns),
        )
        dates = observed["observation_date"].combine_chunks()
        availability = observed[
            "public_availability_date"
        ].combine_chunks()
        if (
            observed.num_rows != 3_752
            or pc.min(dates).as_py() != dt.date(2000, 1, 3)
            or pc.max(dates).as_py() != dt.date(2014, 12, 1)
            or pc.any(pc.greater(availability, dates)).as_py()
        ):
            raise LabelStoreContractError(
                "Bounded Stage 7 factor table differs"
            )
        return observed

    def iter_development_return_batches(
        self,
        *,
        batch_rows: int = 65_536,
    ) -> Iterator[pa.RecordBatch]:
        """Yield return rows only through the authorized 2014-12-01 cutoff."""

        self._require_authenticated()
        if (
            isinstance(batch_rows, bool)
            or not isinstance(batch_rows, int)
            or batch_rows <= 0
        ):
            raise LabelStoreContractError(
                "Stage 7 return batch size must be positive"
            )
        table = self.tables["crsp_security_day_return.parquet"]
        cutoff_date = dt.date(2014, 12, 1)
        cutoff = pa.scalar(cutoff_date, type=pa.date32())
        scanner = ds.dataset(
            self._trusted_path(table.name), format="parquet"
        ).scanner(
            columns=list(table.projected_columns),
            filter=ds.field("calendar_date") <= cutoff,
            batch_size=batch_rows,
            use_threads=False,
        )
        for batch in scanner.to_batches():
            dates = batch.column(
                batch.schema.get_field_index("calendar_date")
            )
            if (
                batch.num_rows
                and pc.max(dates).as_py() > cutoff_date
            ):
                raise LabelStoreContractError(
                    "Post-cutoff return row escaped Stage 7 delivery"
                )
            for name in (
                "return_span_start_date",
                "return_span_end_date",
            ):
                index = batch.schema.get_field_index(name)
                if index < 0:
                    continue
                values = batch.column(index)
                violation = pc.greater(values, cutoff)
                if pc.any(pc.fill_null(violation, False)).as_py():
                    raise LabelStoreContractError(
                        "Post-cutoff return span escaped Stage 7 delivery"
                    )
            yield batch


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text("utf-8"))
    if not isinstance(value, dict):
        raise LabelStoreContractError(
            f"Expected Stage 7 JSON object: {path}"
        )
    return value


def _toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (PROJECT_ROOT / path).resolve()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def _schema_sha256(schema: pa.Schema) -> str:
    return hashlib.sha256(schema.serialize().to_pybytes()).hexdigest()


def _stat_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
        int(value.st_mode),
        int(value.st_nlink),
        int(value.st_uid),
        int(value.st_gid),
    )


def _output_receipts(
    manifest: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    candidates: list[Mapping[str, Any]] = []
    outputs = manifest.get("outputs")
    if isinstance(outputs, list):
        candidates.extend(
            item for item in outputs if isinstance(item, dict)
        )
    elif isinstance(outputs, dict):
        for key, item in outputs.items():
            if isinstance(item, dict) and "files" not in item:
                candidates.append({"table_key": key, **item})
    output = manifest.get("output")
    if isinstance(output, dict):
        candidates.append(output)
    receipts: dict[str, dict[str, Any]] = {}
    for item in candidates:
        path = item.get("path")
        if not isinstance(path, str) or not path.endswith(".parquet"):
            continue
        name = Path(path).name
        if name in receipts:
            raise LabelStoreContractError(
                f"Duplicate upstream Stage 7 Parquet basename: {name}"
            )
        receipts[name] = dict(item)
    return receipts


def _relocated_path(
    value: str,
    *,
    database_root: Path,
    historical_root: Path,
    interim_path: str | None,
) -> Path:
    raw = Path(value).expanduser()
    if raw.is_absolute():
        try:
            relative = raw.relative_to(historical_root)
        except ValueError:
            try:
                relative = raw.relative_to(database_root)
            except ValueError as exc:
                raise LabelStoreContractError(
                    "Stage 7 input is outside authenticated database roots: "
                    f"{value}"
                ) from exc
        candidate = database_root / relative
    elif len(raw.parts) > 1:
        candidate = database_root / raw
    elif interim_path is not None:
        candidate = database_root / interim_path / raw
    else:
        raise LabelStoreContractError(
            f"Cannot resolve Stage 7 upstream output: {value}"
        )
    return _anchored_regular_path(
        candidate,
        trusted_root=database_root,
        label="relocated Stage 7 input",
    )


def _anchored_regular_path(
    candidate: Path,
    *,
    trusted_root: Path,
    label: str,
) -> Path:
    """Reject lexical escapes and every symlink component before resolving."""

    root = Path(os.path.abspath(trusted_root.expanduser()))
    raw = Path(os.path.abspath(candidate.expanduser()))
    try:
        relative = raw.relative_to(root)
    except ValueError as exc:
        raise LabelStoreContractError(
            f"{label} escaped the active database root"
        ) from exc
    if root.is_symlink():
        raise LabelStoreContractError(
            f"Symlinked Stage 7 trusted root is prohibited: {root}"
        )
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise LabelStoreContractError(
                f"Symlinked Stage 7 input component is prohibited: {current}"
            )
    try:
        resolved = raw.resolve(strict=True)
    except OSError as exc:
        raise LabelStoreContractError(
            f"Missing Stage 7 input: {raw}"
        ) from exc
    try:
        resolved.relative_to(root.resolve(strict=True))
    except ValueError as exc:
        raise LabelStoreContractError(
            f"{label} resolved outside the active database root"
        ) from exc
    if not resolved.is_file():
        raise LabelStoreContractError(
            f"Stage 7 input is not a regular file: {resolved}"
        )
    return resolved


def _feature_store_receipt(
    policy: Mapping[str, Any],
    *,
    database_root: Path,
) -> tuple[Path, Mapping[str, Any]]:
    manifest_path = _project_path(
        policy["verified_feature_store"]["primary_manifest"]
    )
    manifest = _json(manifest_path)
    interim = manifest.get("outputs", {}).get("interim")
    if not isinstance(interim, dict):
        raise LabelStoreContractError(
            "Stage 6 interim output receipt is missing"
        )
    files = interim.get("files")
    if not isinstance(files, dict):
        raise LabelStoreContractError(
            "Stage 6 interim closed file set is missing"
        )
    receipt = files.get("feature_store_raw.parquet")
    if not isinstance(receipt, dict):
        raise LabelStoreContractError(
            "Stage 6 raw feature receipt is missing"
        )
    relative = Path(str(interim.get("path", "")))
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or not relative.parts
    ):
        raise LabelStoreContractError(
            "Unsafe Stage 6 raw-feature namespace"
        )
    candidate = (
        database_root / relative / "feature_store_raw.parquet"
    )
    path = _anchored_regular_path(
        candidate,
        trusted_root=database_root,
        label="Stage 6 raw feature input",
    )
    return path, receipt


def _upstream_receipt(
    semantic: Mapping[str, Any],
    *,
    gate: str,
    name: str,
    database_root: Path,
    historical_root: Path,
) -> tuple[Path, Mapping[str, Any]]:
    specification = semantic["upstream"][gate]
    manifest = _json(
        _project_path(specification["primary_manifest"])
    )
    receipts = _output_receipts(manifest)
    try:
        receipt = receipts[name]
    except KeyError as exc:
        raise LabelStoreContractError(
            f"Stage 7 upstream receipt is missing: {name}"
        ) from exc
    interim_path = manifest.get("interim_path")
    if interim_path is not None and not isinstance(interim_path, str):
        raise LabelStoreContractError(
            f"Invalid Stage 7 upstream namespace: {gate}"
        )
    path = _relocated_path(
        str(receipt["path"]),
        database_root=database_root,
        historical_root=historical_root,
        interim_path=interim_path,
    )
    return path, receipt


_GATE_TO_UPSTREAM = {
    "verified_calendar": "calendar",
    "verified_return_reconciliation": "return_reconciliation",
    "verified_signal_universe": "signal_universe",
    "verified_auxiliary": "auxiliary",
}


def _resolve_label_input_catalog(
    policy_path: Path = DEFAULT_POLICY,
    *,
    authenticate_hashes: bool,
) -> LabelStoreInputs:
    """Internal resolver; unauthenticated catalogs cannot deliver rows."""

    contract = validate_label_store_contract(policy_path)
    policy = _toml(policy_path)
    rules = _json(_project_path(policy["physical_rules"]["path"]))
    semantic = _toml(
        _project_path(policy["semantic_contract"]["policy_path"])
    )
    declared_root = Path(policy["database"]["root"]).expanduser()
    if declared_root.is_symlink():
        raise LabelStoreContractError(
            "Symlinked Stage 7 database root is prohibited"
        )
    database_root = declared_root.resolve(strict=True)
    historical_root = Path(
        semantic["database"]["historical_root"]
    ).resolve()
    relocation = _json(
        _project_path(policy["database"]["relocation_receipt"])
    )
    if (
        relocation.get("new_database_root")
        != database_root.as_posix()
        or relocation.get("old_database_root")
        != historical_root.as_posix()
    ):
        raise LabelStoreContractError(
            "Stage 7 relocation authority differs"
        )

    tables: dict[str, LabelInputTable] = {}
    for name in contract["input_names"]:
        expected = rules["input_allowlist"][name]
        gate = str(expected["gate"])
        if gate == "verified_stage_06":
            path, receipt = _feature_store_receipt(
                policy, database_root=database_root
            )
        else:
            try:
                upstream_gate = _GATE_TO_UPSTREAM[gate]
            except KeyError as exc:
                raise LabelStoreContractError(
                    f"Unknown Stage 7 input gate: {gate}"
                ) from exc
            path, receipt = _upstream_receipt(
                semantic,
                gate=upstream_gate,
                name=name,
                database_root=database_root,
                historical_root=historical_root,
            )

        before_stat = path.stat()
        observed_size = before_stat.st_size
        manifest_size = receipt.get("size_bytes")
        expected_hash = str(expected["sha256"])
        if (
            not isinstance(manifest_size, int)
            or observed_size != manifest_size
            or receipt.get("sha256") != expected_hash
        ):
            raise LabelStoreContractError(
                f"Stage 7 input receipt differs: {name}"
            )
        if authenticate_hashes:
            if _sha256_file(path) != expected_hash:
                raise LabelStoreContractError(
                    f"Stage 7 input content hash differs: {name}"
                )
        after_hash_stat = path.stat()
        if _stat_identity(before_stat) != _stat_identity(after_hash_stat):
            raise LabelStoreContractError(
                f"Stage 7 input changed during authentication: {name}"
            )
        parquet = pq.ParquetFile(path)
        rows = parquet.metadata.num_rows
        if rows != int(expected["rows"]):
            raise LabelStoreContractError(
                f"Stage 7 input row count differs: {name}"
            )
        schema_hash = _schema_sha256(parquet.schema_arrow)
        if schema_hash != expected["schema_sha256"]:
            raise LabelStoreContractError(
                f"Stage 7 input schema differs: {name}"
            )
        projection = tuple(str(value) for value in expected[
            "projected_columns"
        ])
        if not set(projection).issubset(parquet.schema_arrow.names):
            raise LabelStoreContractError(
                f"Stage 7 input projection differs: {name}"
            )
        tables[name] = LabelInputTable(
            gate=gate,
            name=name,
            _path=path,
            sha256=expected_hash,
            schema_sha256=schema_hash,
            size_bytes=observed_size,
            rows=rows,
            row_groups=parquet.metadata.num_row_groups,
            projected_columns=projection,
            stat_identity=_stat_identity(after_hash_stat),
        )

    if tuple(tables) != contract["input_names"] or len(tables) != 6:
        raise LabelStoreContractError(
            "Stage 7 resolved catalog differs from closed allowlist"
        )
    return LabelStoreInputs(
        database_root=database_root,
        tables=tables,
        hashes_authenticated=authenticate_hashes,
    )


def resolve_label_input_catalog(
    policy_path: Path = DEFAULT_POLICY,
) -> LabelStoreInputs:
    """Hash-authenticate and return exactly the six Stage 7 inputs."""

    return _resolve_label_input_catalog(
        policy_path,
        authenticate_hashes=True,
    )


def reauthenticate_label_inputs(inputs: LabelStoreInputs) -> None:
    """Recheck stable identity, full hashes, schemas, and rows in place."""

    inputs._require_authenticated()
    for name, table in inputs.tables.items():
        path = inputs._trusted_path(name)
        before = path.stat()
        if _stat_identity(before) != table.stat_identity:
            raise LabelStoreContractError(
                f"Stage 7 input changed before reauthentication: {name}"
            )
        if _sha256_file(path) != table.sha256:
            raise LabelStoreContractError(
                f"Stage 7 input hash changed: {name}"
            )
        after = path.stat()
        if (
            _stat_identity(before) != _stat_identity(after)
            or _stat_identity(after) != table.stat_identity
        ):
            raise LabelStoreContractError(
                f"Stage 7 input changed during reauthentication: {name}"
            )
        parquet = pq.ParquetFile(path)
        if (
            parquet.metadata.num_rows != table.rows
            or parquet.metadata.num_row_groups != table.row_groups
            or _schema_sha256(parquet.schema_arrow)
            != table.schema_sha256
        ):
            raise LabelStoreContractError(
                f"Stage 7 input metadata changed: {name}"
            )
