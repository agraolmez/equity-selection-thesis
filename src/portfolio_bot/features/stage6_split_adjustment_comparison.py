"""Pinned scientific comparison of the pre/post split-adjustment Stage 6 builds.

The gate is intentionally narrow: it reads only authenticated Stage 6 feature
artifacts from the development period and never opens labels or later outcomes.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import io
import json
import math
import os
import stat
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


OLD_BUILD_ID = "phase03-features-780718caab15c6156d396a2b"
NEW_BUILD_ID = "phase03-features-fda21f0e51b8276c10246f9f"
CONTRACT_ID = "phase03-feature-label-split-v1"
POLICY_ID = "phase03-feature-store-policy-v1"
EXPECTED_ROWS = 1_263_048
EXPECTED_SIGNAL_DATES = 178
FIRST_SIGNAL_DATE = dt.date(2000, 1, 31)
LAST_SIGNAL_DATE = dt.date(2014, 10, 31)
FIRST_PROHIBITED_DATE = dt.date(2015, 1, 1)

AUDIT_FIELDS = (
    pa.field("signal_date", pa.date32(), nullable=False),
    pa.field("permno", pa.int64(), nullable=False),
    pa.field("permco", pa.int64(), nullable=False),
    pa.field("long_eligible", pa.bool_(), nullable=False),
    pa.field("short_eligible", pa.bool_(), nullable=False),
    pa.field("primary_exchange", pa.string(), nullable=True),
    pa.field("share_type", pa.string(), nullable=True),
)
AUDIT_COLUMNS = tuple(field.name for field in AUDIT_FIELDS)
KEY_COLUMNS = ("signal_date", "permno")
SPLIT_SENSITIVE_COLUMNS = (
    "price_to_ma_21d",
    "price_to_ma_63d",
    "price_to_ma_126d",
    "price_to_ma_252d",
    "distance_to_high_252d",
    "log_price_trend_slope_63d",
    "log_price_trend_r2_63d",
    "log_price_trend_slope_126d",
    "log_price_trend_r2_126d",
    "log_price_trend_slope_252d",
    "log_price_trend_r2_252d",
)

_MANIFEST_KEYS = {
    "schema_version", "contract_id", "policy_id", "build_id", "status",
    "bindings", "inputs", "outputs", "summary", "authorizations",
}
_OUTPUT_KEYS = {"path", "success_sha256", "files"}
_FILE_KEYS = {
    "path", "sha256", "size_bytes", "rows", "row_groups", "schema_sha256",
}
_VERIFICATION_KEYS = {
    "schema_version", "contract_id", "policy_id", "build_id",
    "verification_id", "status", "primary_manifest_sha256", "checks",
    "differences_sha256", "output_hashes_sha256", "authorizations",
}
_DICTIONARY_COLUMNS = (
    "feature_ordinal", "feature_name", "family", "kind", "source_set",
    "unit", "formula_id", "parameters_json",
)


@dataclass(frozen=True)
class BuildPin:
    build_id: str
    manifest_sha256: str
    verification_id: str
    verification_sha256: str


OLD_PIN = BuildPin(
    OLD_BUILD_ID,
    "5324dfef6accbf4c460bbb193f049ad77139c64a3eda4ba88ba61df47a1b7729",
    "verify-8d40e532b4e815cf",
    "92502275e47447a3644a4be9ff8d679f98693869484a261d700db04fe3b27d6d",
)
NEW_PIN = BuildPin(
    NEW_BUILD_ID,
    "96ab135c0628054699fa3f27a76abd1cb30f3e3e46279a84052a29c895b51375",
    "verify-350915782d3ef12e",
    "0a7636aaf076d9b338023379c3cd5c7340cf4b852a8336813b1428275bace964",
)


class Stage6ComparisonError(RuntimeError):
    """Raised when the pinned scientific comparison cannot pass safely."""


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False) + "\n").encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def _schema_sha256(schema: pa.Schema) -> str:
    return hashlib.sha256(schema.serialize().to_pybytes()).hexdigest()


def _regular_file(path: Path, label: str) -> Path:
    path = Path(os.path.abspath(path.expanduser()))
    try:
        observed = os.lstat(path)
    except OSError as exc:
        raise Stage6ComparisonError(f"missing {label}: {path}") from exc
    if not stat.S_ISREG(observed.st_mode) or stat.S_ISLNK(observed.st_mode):
        raise Stage6ComparisonError(f"unsafe {label}: {path}")
    return path


def _canonical_object(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    payload = _regular_file(path, label).read_bytes()
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise Stage6ComparisonError(f"invalid JSON in {label}") from exc
    if not isinstance(value, dict) or payload != canonical_json_bytes(value):
        raise Stage6ComparisonError(f"noncanonical {label}")
    return value, payload


def _count_true(array: pa.Array | pa.ChunkedArray) -> int:
    value = pc.sum(pc.cast(pc.fill_null(array, False), pa.int64())).as_py()
    return 0 if value is None else int(value)


def _receipt_for_file(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": path.name,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "rows": None,
        "row_groups": None,
        "schema_sha256": None,
    }
    if path.suffix == ".parquet":
        parquet = pq.ParquetFile(path)
        result.update(
            rows=parquet.metadata.num_rows,
            row_groups=parquet.metadata.num_row_groups,
            schema_sha256=_schema_sha256(parquet.schema_arrow),
        )
    return result


def _authenticate_build(
    *, project_root: Path, database_root: Path, pin: BuildPin,
) -> dict[str, Any]:
    artifact_root = project_root / "artifacts/phase_03_features"
    manifest_path = artifact_root / "feature_store_builds" / pin.build_id / "manifest.json"
    manifest, manifest_bytes = _canonical_object(manifest_path, "Stage 6 manifest")
    if hashlib.sha256(manifest_bytes).hexdigest() != pin.manifest_sha256:
        raise Stage6ComparisonError(f"pinned manifest hash differs: {pin.build_id}")
    if set(manifest) != _MANIFEST_KEYS or (
        manifest.get("schema_version") != 1
        or manifest.get("contract_id") != CONTRACT_ID
        or manifest.get("policy_id") != POLICY_ID
        or manifest.get("build_id") != pin.build_id
        or manifest.get("status") != "PRIMARY_PASSED_PENDING_INDEPENDENT_VERIFICATION"
    ):
        raise Stage6ComparisonError(f"manifest contract differs: {pin.build_id}")

    verification_path = (
        artifact_root / "feature_store_verifications" / pin.build_id
        / pin.verification_id / "independent_check.json"
    )
    verification, verification_bytes = _canonical_object(
        verification_path, "Stage 6 independent verification"
    )
    if hashlib.sha256(verification_bytes).hexdigest() != pin.verification_sha256:
        raise Stage6ComparisonError(f"pinned verification hash differs: {pin.build_id}")
    if set(verification) != _VERIFICATION_KEYS or (
        verification.get("build_id") != pin.build_id
        or verification.get("verification_id") != pin.verification_id
        or verification.get("status") != "VERIFIED_PASSED"
        or verification.get("primary_manifest_sha256") != pin.manifest_sha256
        or verification.get("checks", {}).get("fail_count") != 0
        or verification.get("authorizations", {}).get(
            "independent_verification_completed"
        ) is not True
    ):
        raise Stage6ComparisonError(f"verification contract differs: {pin.build_id}")
    differences_path = verification_path.with_name("differences.json")
    differences = _regular_file(differences_path, "verification differences").read_bytes()
    if differences != canonical_json_bytes([]) or hashlib.sha256(differences).hexdigest() != verification["differences_sha256"]:
        raise Stage6ComparisonError(f"verification differences are not empty: {pin.build_id}")

    output_hashes: dict[str, dict[str, str]] = {}
    authenticated: dict[str, Path] = {}
    for kind in ("interim", "processed", "quality"):
        specification = manifest["outputs"].get(kind)
        if not isinstance(specification, Mapping) or set(specification) != _OUTPUT_KEYS:
            raise Stage6ComparisonError(f"manifest {kind} output differs")
        output_hashes[kind] = {
            name: str(receipt["sha256"])
            for name, receipt in sorted(specification["files"].items())
        }
    output_hashes_sha = hashlib.sha256(canonical_json_bytes(output_hashes)).hexdigest()
    if output_hashes_sha != verification["output_hashes_sha256"]:
        raise Stage6ComparisonError(f"verification output binding differs: {pin.build_id}")

    required = {
        "raw": ("interim", "feature_store_raw.parquet"),
        "dictionary": ("processed", "feature_dictionary.parquet"),
        "vocabulary": ("processed", "category_vocabulary.parquet"),
    }
    file_receipts: dict[str, dict[str, Any]] = {}
    for label, (kind, name) in required.items():
        specification = manifest["outputs"][kind]
        receipt = specification["files"].get(name)
        if not isinstance(receipt, Mapping) or set(receipt) != _FILE_KEYS:
            raise Stage6ComparisonError(f"missing exact {label} receipt: {pin.build_id}")
        relative = Path(str(specification["path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise Stage6ComparisonError(f"unsafe output namespace: {pin.build_id}")
        expected_tail = Path(kind) / "phase_03_features" / CONTRACT_ID / "feature_store_builds" / pin.build_id
        if relative != expected_tail:
            raise Stage6ComparisonError(f"output namespace differs: {pin.build_id}/{kind}")
        path = _regular_file(database_root / relative / name, f"{pin.build_id} {label}")
        observed = _receipt_for_file(path)
        if dict(receipt) != observed:
            raise Stage6ComparisonError(f"authenticated {label} bytes differ: {pin.build_id}")
        authenticated[label] = path
        file_receipts[label] = observed
    return {
        "pin": pin,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "verification_path": verification_path,
        "files": authenticated,
        "file_receipts": file_receipts,
    }


def _dictionary_rows(path: Path) -> list[dict[str, Any]]:
    table = pq.read_table(path)
    if tuple(table.column_names) != _DICTIONARY_COLUMNS or table.num_rows != 163:
        raise Stage6ComparisonError("feature dictionary shape differs")
    rows = table.to_pylist()
    if [row["feature_ordinal"] for row in rows] != list(range(1, 164)):
        raise Stage6ComparisonError("feature ordinals differ")
    names = [str(row["feature_name"]) for row in rows]
    if len(names) != len(set(names)):
        raise Stage6ComparisonError("duplicate feature names")
    return rows


def _validate_dictionaries(old_path: Path, new_path: Path) -> list[dict[str, Any]]:
    old = _dictionary_rows(old_path)
    new = _dictionary_rows(new_path)
    names = [str(row["feature_name"]) for row in new]
    if [str(row["feature_name"]) for row in old] != names:
        raise Stage6ComparisonError("old/new feature dictionary names differ")
    if set(SPLIT_SENSITIVE_COLUMNS) - set(names) or len(names) - len(SPLIT_SENSITIVE_COLUMNS) != 152:
        raise Stage6ComparisonError("152/11 feature partition differs")
    sensitive = set(SPLIT_SENSITIVE_COLUMNS)
    for left, right in zip(old, new, strict=True):
        name = str(left["feature_name"])
        if name not in sensitive:
            if left != right:
                raise Stage6ComparisonError(f"unaffected dictionary row differs: {name}")
            continue
        for key in _DICTIONARY_COLUMNS:
            if key == "source_set":
                continue
            if left[key] != right[key]:
                raise Stage6ComparisonError(f"split-sensitive dictionary metadata differs: {name}/{key}")
        if left["source_set"] != "crsp_market" or right["source_set"] != "crsp_price_returns":
            raise Stage6ComparisonError(f"split-sensitive source transition differs: {name}")
    return new


def _category_domains(path: Path) -> dict[str, set[str]]:
    table = pq.read_table(path, columns=["feature_name", "category_value"])
    result: dict[str, set[str]] = {}
    for row in table.to_pylist():
        result.setdefault(str(row["feature_name"]), set()).add(str(row["category_value"]))
    return result


class _Cursor:
    def __init__(self, path: Path, batch_rows: int) -> None:
        self._iterator = pq.ParquetFile(path).iter_batches(
            batch_size=batch_rows, use_threads=False
        )
        self._batch: pa.RecordBatch | None = None
        self._offset = 0

    def take(self, rows: int) -> pa.Table:
        batches: list[pa.RecordBatch] = []
        remaining = rows
        while remaining:
            if self._batch is None or self._offset == self._batch.num_rows:
                try:
                    self._batch = next(self._iterator)
                except StopIteration:
                    break
                self._offset = 0
            take = min(remaining, self._batch.num_rows - self._offset)
            batches.append(self._batch.slice(self._offset, take))
            self._offset += take
            remaining -= take
        return pa.Table.from_batches(batches) if batches else pa.table({})


def null_aware_counts(left: pa.Array, right: pa.Array) -> dict[str, int]:
    """Return exact (no-tolerance) equality and null-transition counts."""
    if len(left) != len(right) or left.type != right.type:
        raise Stage6ComparisonError("array shape/type differs")
    old_null = pc.is_null(left)
    new_null = pc.is_null(right)
    both_null = pc.and_(old_null, new_null)
    try:
        equal_value = pc.fill_null(pc.equal(left, right), False)
        equal_rows = _count_true(pc.or_(both_null, equal_value))
        null_to_value = _count_true(pc.and_(old_null, pc.invert(new_null)))
        value_to_null = _count_true(pc.and_(pc.invert(old_null), new_null))
    except pa.ArrowNotImplementedError:
        pairs = zip(left.to_pylist(), right.to_pylist(), strict=True)
        equal_rows = null_to_value = value_to_null = 0
        for old, new in pairs:
            equal_rows += int(old == new)
            null_to_value += int(old is None and new is not None)
            value_to_null += int(old is not None and new is None)
    changed_rows = len(left) - equal_rows
    return {
        "equal_rows": equal_rows,
        "changed_rows": changed_rows,
        "old_nulls": left.null_count,
        "new_nulls": right.null_count,
        "null_to_value": null_to_value,
        "value_to_null": value_to_null,
        "value_to_value": changed_rows - null_to_value - value_to_null,
    }


def _domain_violations(name: str, values: pa.Array) -> int:
    finite_bad = _count_true(pc.invert(pc.fill_null(pc.is_finite(values), True)))
    if name.startswith("price_to_ma_"):
        return finite_bad + _count_true(pc.fill_null(pc.less_equal(values, -1.0), False))
    if name == "distance_to_high_252d":
        below = pc.fill_null(pc.less(values, -1.0), False)
        above = pc.fill_null(pc.greater(values, 0.0), False)
        return finite_bad + _count_true(pc.or_(below, above))
    if "_r2_" in name:
        below = pc.fill_null(pc.less(values, 0.0), False)
        above = pc.fill_null(pc.greater(values, 1.0), False)
        return finite_bad + _count_true(pc.or_(below, above))
    return finite_bad


def _stream_compare(
    old_path: Path,
    new_path: Path,
    dictionary: Sequence[Mapping[str, Any]],
    category_domains: Mapping[str, set[str]],
    *,
    batch_rows: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    old_file, new_file = pq.ParquetFile(old_path), pq.ParquetFile(new_path)
    if old_file.schema_arrow != new_file.schema_arrow:
        raise Stage6ComparisonError("old/new raw schemas differ")
    schema = new_file.schema_arrow
    if tuple(schema.field(index) for index in range(7)) != AUDIT_FIELDS:
        raise Stage6ComparisonError("exact Stage 6 audit schema differs")
    feature_names = tuple(str(row["feature_name"]) for row in dictionary)
    if tuple(schema.names) != AUDIT_COLUMNS + feature_names:
        raise Stage6ComparisonError("raw schema is not exact audit-plus-dictionary order")
    if old_file.metadata.num_rows != EXPECTED_ROWS or new_file.metadata.num_rows != EXPECTED_ROWS:
        raise Stage6ComparisonError("raw row count differs")

    classes = {
        name: ("key" if name in KEY_COLUMNS else "audit") for name in AUDIT_COLUMNS
    }
    sensitive = set(SPLIT_SENSITIVE_COLUMNS)
    classes.update({name: "split_sensitive_feature" if name in sensitive else "unaffected_feature" for name in feature_names})
    totals = {
        name: {"column_name": name, "column_class": classes[name],
               "rows_compared": 0, "equal_rows": 0, "changed_rows": 0,
               "old_nulls": 0, "new_nulls": 0, "null_to_value": 0,
               "value_to_null": 0, "value_to_value": 0,
               "domain_violations_old": 0, "domain_violations_new": 0}
        for name in schema.names
    }
    kinds = {str(row["feature_name"]): str(row["kind"]) for row in dictionary}
    old_cursor, new_cursor = _Cursor(old_path, batch_rows), _Cursor(new_path, batch_rows)
    rows = key_order_errors = short_without_long = audit_domain_errors = 0
    signal_dates: set[dt.date] = set()
    first: dt.date | None = None
    last: dt.date | None = None
    prior_key: tuple[dt.date, int] | None = None

    while rows < EXPECTED_ROWS:
        old = old_cursor.take(min(batch_rows, EXPECTED_ROWS - rows))
        new = new_cursor.take(min(batch_rows, EXPECTED_ROWS - rows))
        if old.num_rows == 0 or old.num_rows != new.num_rows:
            raise Stage6ComparisonError("raw streams ended at different rows")
        n = old.num_rows
        for name in schema.names:
            left = old[name].combine_chunks()
            right = new[name].combine_chunks()
            counts = null_aware_counts(left, right)
            target = totals[name]
            target["rows_compared"] += n
            for key, value in counts.items():
                target[key] += value

        signals = new["signal_date"].combine_chunks().to_pylist()
        permnos = new["permno"].combine_chunks().to_pylist()
        for signal, permno in zip(signals, permnos, strict=True):
            key = (signal, int(permno))
            key_order_errors += int(prior_key is not None and key <= prior_key)
            prior_key = key
            signal_dates.add(signal)
            first = signal if first is None else min(first, signal)
            last = signal if last is None else max(last, signal)
            audit_domain_errors += int(signal >= FIRST_PROHIBITED_DATE)
        audit_domain_errors += _count_true(pc.less_equal(new["permno"], 0))
        audit_domain_errors += _count_true(pc.less_equal(new["permco"], 0))
        short_without_long += _count_true(pc.and_(new["short_eligible"], pc.invert(new["long_eligible"])))
        for name, allowed in (
            ("primary_exchange", {"A", "B", "N", "Q", "R", "X"}),
            ("share_type", {"AD", "CE", "N/A", "NS", "SB", "UG"}),
        ):
            column = new[name].combine_chunks()
            audit_domain_errors += column.null_count
            audit_domain_errors += _count_true(pc.invert(pc.fill_null(pc.is_in(column, value_set=pa.array(sorted(allowed))), True)))

        for name in feature_names:
            column = new[name].combine_chunks()
            kind = kinds[name]
            if kind in {"continuous_firm", "aggregate_state"}:
                totals[name]["domain_violations_new"] += _domain_violations(name, column)
            elif kind == "binary_firm":
                allowed = pa.array([0, 1], type=column.type)
                valid = pc.or_(pc.is_null(column), pc.is_in(column, value_set=allowed))
                totals[name]["domain_violations_new"] += _count_true(
                    pc.invert(pc.fill_null(valid, False))
                )
            elif kind == "categorical_firm":
                allowed = category_domains.get(name)
                if not allowed:
                    raise Stage6ComparisonError(f"missing category domain: {name}")
                totals[name]["domain_violations_new"] += column.null_count
                # Dynamic vocabularies are intentionally populated only from
                # long-eligible rows, while the audit store also retains
                # normalized categories on ineligible rows.  Their stable raw
                # domain is therefore non-null, non-empty UTF-8 rather than
                # membership in the model-encoding vocabulary.
                totals[name]["domain_violations_new"] += _count_true(
                    pc.fill_null(pc.equal(pc.utf8_length(column), 0), False)
                )
            elif kind == "categorical_multihot_firm":
                allowed = category_domains.get(name)
                if not allowed:
                    raise Stage6ComparisonError(f"missing category domain: {name}")
                totals[name]["domain_violations_new"] += column.null_count
                lengths = pc.list_value_length(column)
                totals[name]["domain_violations_new"] += _count_true(pc.fill_null(pc.less_equal(lengths, 0), False))
                flat = pc.list_flatten(column)
                totals[name]["domain_violations_new"] += flat.null_count
                totals[name]["domain_violations_new"] += _count_true(
                    pc.fill_null(pc.equal(pc.utf8_length(flat), 0), False)
                )
            else:
                raise Stage6ComparisonError(f"unknown feature kind: {kind}")
            if name in sensitive:
                totals[name]["domain_violations_old"] += _domain_violations(name, old[name].combine_chunks())
        rows += n

    if first != FIRST_SIGNAL_DATE or last != LAST_SIGNAL_DATE or len(signal_dates) != EXPECTED_SIGNAL_DATES:
        raise Stage6ComparisonError("development-only signal-date domain differs")
    if key_order_errors or short_without_long or audit_domain_errors:
        raise Stage6ComparisonError("key or audit domain failed")
    records = [totals[name] for name in schema.names]
    unchanged = [row for row in records if row["column_class"] in {"key", "audit", "unaffected_feature"}]
    changed = [row for row in records if row["column_class"] == "split_sensitive_feature"]
    if len(unchanged) != 159 or len(changed) != 11:
        raise Stage6ComparisonError("159/11 raw-column partition differs")
    if any(row["changed_rows"] for row in unchanged):
        names = [row["column_name"] for row in unchanged if row["changed_rows"]]
        raise Stage6ComparisonError(f"change outside split-sensitive allowlist: {names}")
    intended_changes = sum(int(row["changed_rows"]) for row in changed)
    if intended_changes <= 0:
        raise Stage6ComparisonError("no intended split-sensitive value changed")
    invalid_domains = {
        row["column_name"]: {
            "old": row["domain_violations_old"],
            "new": row["domain_violations_new"],
        }
        for row in records
        if row["domain_violations_old"] or row["domain_violations_new"]
    }
    if invalid_domains:
        raise Stage6ComparisonError(
            "raw feature domain violation: "
            + json.dumps(invalid_domains, sort_keys=True)
        )
    summary = {
        "rows_compared": rows,
        "key_columns_exact": 2,
        "audit_nonkey_columns_exact": 5,
        "unaffected_feature_columns_exact": 152,
        "split_sensitive_columns_allowed": 11,
        "split_sensitive_changed_cell_count": intended_changes,
        "first_signal_date": first.isoformat(),
        "last_signal_date": last.isoformat(),
        "signal_date_count": len(signal_dates),
        "key_order_errors": key_order_errors,
        "domain_violations": 0,
    }
    return records, summary


def _csv_bytes(records: Iterable[Mapping[str, Any]]) -> bytes:
    fields = (
        "column_ordinal", "column_name", "column_class", "rows_compared",
        "equal_rows", "changed_rows", "old_nulls", "new_nulls",
        "null_to_value", "value_to_null", "value_to_value",
        "domain_violations_old", "domain_violations_new",
    )
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for ordinal, record in enumerate(records, 1):
        writer.writerow({"column_ordinal": ordinal, **record})
    return output.getvalue().encode("utf-8")


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def compare_stage6_split_adjustment(
    *, project_root: Path, database_root: Path, batch_rows: int = 16_384,
) -> dict[str, Any]:
    if not isinstance(batch_rows, int) or batch_rows <= 0:
        raise Stage6ComparisonError("batch_rows must be positive")
    project_root = Path(os.path.abspath(project_root.expanduser()))
    database_root = Path(os.path.abspath(database_root.expanduser()))
    old = _authenticate_build(project_root=project_root, database_root=database_root, pin=OLD_PIN)
    new = _authenticate_build(project_root=project_root, database_root=database_root, pin=NEW_PIN)
    dictionary = _validate_dictionaries(old["files"]["dictionary"], new["files"]["dictionary"])
    old_vocab, new_vocab = pq.read_table(old["files"]["vocabulary"]), pq.read_table(new["files"]["vocabulary"])
    if not old_vocab.equals(new_vocab):
        raise Stage6ComparisonError("category vocabulary changed")
    records, summary = _stream_compare(
        old["files"]["raw"], new["files"]["raw"], dictionary,
        _category_domains(new["files"]["vocabulary"]), batch_rows=batch_rows,
    )
    csv_payload = _csv_bytes(records)
    csv_sha = hashlib.sha256(csv_payload).hexdigest()
    module_sha = sha256_file(Path(__file__).resolve())
    identity = {
        "old_build_id": OLD_BUILD_ID, "new_build_id": NEW_BUILD_ID,
        "old_manifest_sha256": OLD_PIN.manifest_sha256,
        "new_manifest_sha256": NEW_PIN.manifest_sha256,
        "old_raw_sha256": old["file_receipts"]["raw"]["sha256"],
        "new_raw_sha256": new["file_receipts"]["raw"]["sha256"],
        "column_comparison_sha256": csv_sha,
        "verifier_code_sha256": module_sha,
    }
    comparison_id = "stage06-split-comparison-" + hashlib.sha256(canonical_json_bytes(identity)).hexdigest()[:16]
    destination = (
        project_root / "artifacts/phase_03_features/split_adjustment_comparisons"
        / f"{OLD_BUILD_ID}__{NEW_BUILD_ID}"
    )
    csv_path = destination / "column_comparison.csv"
    json_path = destination / "comparison_receipt.json"
    receipt = {
        "schema_version": 1,
        "comparison_id": comparison_id,
        "status": "PASSED",
        "scope": {
            "data_accessed": "authenticated Stage 6 development feature artifacts only",
            "outcomes_accessed": False,
            "validation_2015_2019_accessed": False,
            "final_2020_2025_accessed": False,
        },
        "old_build": {
            "build_id": OLD_BUILD_ID, "manifest_sha256": OLD_PIN.manifest_sha256,
            "verification_id": OLD_PIN.verification_id,
            "verification_sha256": OLD_PIN.verification_sha256,
            "raw_file_receipt": old["file_receipts"]["raw"],
        },
        "corrected_build": {
            "build_id": NEW_BUILD_ID, "manifest_sha256": NEW_PIN.manifest_sha256,
            "verification_id": NEW_PIN.verification_id,
            "verification_sha256": NEW_PIN.verification_sha256,
            "raw_file_receipt": new["file_receipts"]["raw"],
        },
        "contract": {
            "key_columns": list(KEY_COLUMNS),
            "audit_columns": list(AUDIT_COLUMNS),
            "economic_feature_count": 163,
            "unaffected_feature_count": 152,
            "split_sensitive_columns": list(SPLIT_SENSITIVE_COLUMNS),
            "comparison_semantics": "exact null-aware equality with no numeric tolerance",
        },
        "checks": {
            "pinned_manifests_authenticated": True,
            "pinned_independent_verifications_authenticated": True,
            "compared_files_rehashed_against_manifests": True,
            "exact_key_spine_and_audit_columns": True,
            "all_152_unaffected_features_exact": True,
            "changes_confined_to_11_split_sensitive_columns": True,
            "at_least_one_intended_change": True,
            "domains_valid": True,
            "development_period_only": True,
        },
        "summary": summary,
        "column_comparison": {
            "path": csv_path.relative_to(project_root).as_posix(),
            "rows": len(records), "sha256": csv_sha,
        },
        "verifier_code_sha256": module_sha,
    }
    _atomic_write(csv_path, csv_payload)
    _atomic_write(json_path, canonical_json_bytes(receipt))
    return {**receipt, "receipt_path": json_path.as_posix(), "csv_path": csv_path.as_posix()}


__all__ = [
    "Stage6ComparisonError", "SPLIT_SENSITIVE_COLUMNS",
    "compare_stage6_split_adjustment", "null_aware_counts",
]
