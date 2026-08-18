"""Frozen raw-value/lineage missingness semantics for Stage 11.

Stage 6 intentionally stores categorical missing states as non-null UNKNOWN
placeholders.  Numeric and binary missing states remain physical nulls.  This
module reproduces that exact frozen rule for candidate construction and for
both primary and independent raw-to-lineage audits.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from portfolio_bot.features.feature_store_contract import expanded_features
from portfolio_bot.features.feature_lineage import VALUE_STATUS_SET
from portfolio_bot.validation.validation_data_contract import PROJECT_ROOT


_REGISTRY_PATH = PROJECT_ROOT / "configs/phase_03_features/feature_registry_v1.json"


class ValidationFeatureSemanticsError(RuntimeError):
    """Raised when frozen raw-value/lineage semantics differ."""


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationFeatureSemanticsError(message)


@lru_cache(maxsize=1)
def frozen_feature_kinds() -> Mapping[str, str]:
    _check(
        _REGISTRY_PATH.is_file() and not _REGISTRY_PATH.is_symlink(),
        "Frozen feature registry is missing or unsafe",
    )
    value = json.loads(_REGISTRY_PATH.read_text("utf-8"))
    _check(isinstance(value, dict), "Frozen feature registry is not an object")
    features = expanded_features(value)
    result = {str(item["name"]): str(item["kind"]) for item in features}
    _check(len(result) == len(features) == 163, "Frozen feature-kind map differs")
    return MappingProxyType(result)


def semantic_value_is_missing(feature_name: str, value: object) -> bool:
    """Apply the exact Stage 6 scalar missingness rule."""

    try:
        kind = frozen_feature_kinds()[feature_name]
    except KeyError as exc:
        raise ValidationFeatureSemanticsError(
            f"Unknown frozen feature: {feature_name}"
        ) from exc
    if kind == "categorical_firm":
        return value is None or (
            isinstance(value, str) and value.startswith("UNKNOWN")
        )
    if kind == "categorical_multihot_firm":
        return value is None or value == ["UNKNOWN"] or value == ("UNKNOWN",)
    return value is None


def semantic_missing_mask(feature_name: str, values: pa.Array) -> np.ndarray:
    """Vectorized exact Stage 6 missingness rule for one raw column."""

    kind = frozen_feature_kinds().get(feature_name)
    _check(kind is not None, f"Unknown frozen feature: {feature_name}")
    if kind == "categorical_firm":
        unknown = pc.fill_null(pc.starts_with(values, "UNKNOWN"), False)
        return pc.or_(pc.is_null(values), unknown).to_numpy(zero_copy_only=False)
    if kind == "categorical_multihot_firm":
        return np.fromiter(
            (value is None or value == ["UNKNOWN"] for value in values.to_pylist()),
            dtype=np.bool_,
            count=len(values),
        )
    return values.is_null().to_numpy(zero_copy_only=False)


class _BatchCursor:
    def __init__(
        self,
        path: Path,
        *,
        columns: Sequence[str],
        batch_size: int,
    ) -> None:
        self._iterator = iter(
            pq.ParquetFile(path).iter_batches(
                batch_size=batch_size,
                columns=list(columns),
                use_threads=False,
            )
        )
        self._active: pa.RecordBatch | None = None
        self._offset = 0

    def take(self, rows: int) -> pa.Table:
        parts: list[pa.RecordBatch] = []
        remaining = rows
        while remaining:
            if self._active is None:
                try:
                    self._active = next(self._iterator)
                    self._offset = 0
                except StopIteration:
                    break
            available = self._active.num_rows - self._offset
            width = min(available, remaining)
            parts.append(self._active.slice(self._offset, width))
            self._offset += width
            remaining -= width
            if self._offset == self._active.num_rows:
                self._active = None
        return pa.Table.from_batches(parts).combine_chunks() if parts else pa.table({})

    def has_more(self) -> bool:
        if self._active is not None and self._offset < self._active.num_rows:
            return True
        try:
            self._active = next(self._iterator)
            self._offset = 0
        except StopIteration:
            return False
        return self._active.num_rows > 0


def _count_true(value: pa.Array | pa.ChunkedArray) -> int:
    scalar = pc.sum(pc.cast(pc.fill_null(value, False), pa.int64())).as_py()
    return int(scalar or 0)


@dataclass(frozen=True, slots=True)
class RawLineageSemanticAudit:
    raw_rows: int
    lineage_rows: int
    key_or_feature_differences: int
    unknown_statuses: int
    value_status_mismatches: int
    timing_violations: int

    @property
    def difference_count(self) -> int:
        return (
            self.key_or_feature_differences
            + self.unknown_statuses
            + self.value_status_mismatches
            + self.timing_violations
        )


def audit_raw_lineage_semantics(
    *,
    raw_path: Path,
    lineage_path: Path,
    feature_names: Sequence[str],
    allowed_signal_dates: Iterable[dt.date],
    batch_rows: int = 2_048,
) -> RawLineageSemanticAudit:
    """Stream-check ordered keys, status, timing, and raw-value consistency."""

    names = tuple(feature_names)
    allowed = frozenset(allowed_signal_dates)
    _check(bool(names) and bool(allowed) and batch_rows > 0, "Semantic audit scope is empty")
    raw_columns = ("signal_date", "permno", *names)
    lineage_columns = (
        "signal_date",
        "permno",
        "feature_name",
        "max_observation_date",
        "max_public_availability_date",
        "stale_after_date",
        "value_status",
    )
    raw_cursor = _BatchCursor(raw_path, columns=raw_columns, batch_size=batch_rows)
    lineage_cursor = _BatchCursor(
        lineage_path,
        columns=lineage_columns,
        batch_size=batch_rows * len(names),
    )
    totals = {
        "raw_rows": 0,
        "lineage_rows": 0,
        "key_or_feature_differences": 0,
        "unknown_statuses": 0,
        "value_status_mismatches": 0,
        "timing_violations": 0,
    }
    while True:
        raw = raw_cursor.take(batch_rows)
        if raw.num_rows == 0:
            break
        needed = raw.num_rows * len(names)
        lineage = lineage_cursor.take(needed)
        _check(lineage.num_rows == needed, "Lineage ended before raw feature rows")
        totals["raw_rows"] += raw.num_rows
        totals["lineage_rows"] += lineage.num_rows

        raw_signals = raw["signal_date"].chunk(0)
        raw_permnos = raw["permno"].chunk(0)
        _check(
            all(value.as_py() in allowed for value in raw_signals),
            "Raw semantic audit escaped its signal scope",
        )
        expected_signal = pa.array(
            np.repeat(raw_signals.to_numpy(zero_copy_only=False), len(names)),
            type=pa.date32(),
        )
        expected_permno = pa.array(
            np.repeat(raw_permnos.to_numpy(zero_copy_only=False), len(names)),
            type=pa.int64(),
        )
        expected_feature = pa.array(list(names) * raw.num_rows, type=pa.string())
        for actual, expected in (
            (lineage["signal_date"].chunk(0), expected_signal),
            (lineage["permno"].chunk(0), expected_permno),
            (lineage["feature_name"].chunk(0), expected_feature),
        ):
            totals["key_or_feature_differences"] += _count_true(
                pc.invert(pc.fill_null(pc.equal(actual, expected), False))
            )

        status = lineage["value_status"].chunk(0)
        known = pc.is_in(status, value_set=pa.array(sorted(VALUE_STATUS_SET)))
        totals["unknown_statuses"] += _count_true(pc.invert(pc.fill_null(known, False)))
        present = pc.fill_null(pc.equal(status, "PRESENT"), False)
        semantic_missing = np.stack(
            [semantic_missing_mask(name, raw[name].chunk(0)) for name in names],
            axis=1,
        ).reshape(-1)
        present_numpy = present.to_numpy(zero_copy_only=False)
        totals["value_status_mismatches"] += int(
            np.count_nonzero(semantic_missing == present_numpy)
        )

        observation = lineage["max_observation_date"].chunk(0)
        availability = lineage["max_public_availability_date"].chunk(0)
        stale_after = lineage["stale_after_date"].chunk(0)
        valid_present_timing = pc.and_kleene(
            pc.and_kleene(
                pc.and_kleene(
                    pc.invert(pc.is_null(observation)),
                    pc.less_equal(observation, expected_signal),
                ),
                pc.and_kleene(
                    pc.invert(pc.is_null(availability)),
                    pc.less_equal(availability, expected_signal),
                ),
            ),
            pc.or_kleene(
                pc.is_null(stale_after),
                pc.less_equal(expected_signal, stale_after),
            ),
        )
        missing_has_dates = pc.or_(
            pc.or_(pc.invert(pc.is_null(observation)), pc.invert(pc.is_null(availability))),
            pc.invert(pc.is_null(stale_after)),
        )
        bad_timing = pc.or_(
            pc.and_(present, pc.invert(pc.fill_null(valid_present_timing, False))),
            pc.and_(pc.invert(present), missing_has_dates),
        )
        totals["timing_violations"] += _count_true(bad_timing)

    _check(not raw_cursor.has_more(), "Raw semantic cursor has extra rows")
    _check(not lineage_cursor.has_more(), "Lineage has rows beyond the raw feature store")
    return RawLineageSemanticAudit(**totals)


__all__ = [
    "RawLineageSemanticAudit",
    "ValidationFeatureSemanticsError",
    "audit_raw_lineage_semantics",
    "frozen_feature_kinds",
    "semantic_missing_mask",
    "semantic_value_is_missing",
]
