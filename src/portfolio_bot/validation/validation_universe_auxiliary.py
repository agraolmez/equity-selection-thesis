"""Stage 11 universe and auxiliary-data scope adapters.

Stage 2 is intentionally frozen at the development boundary.  This module
does not alter it.  Instead, it supplies an exact validation calendar to the
already-frozen universe computation and temporarily narrows the existing
auxiliary normalizers to the Stage 11 knowledge boundary.  The one November
2014 signal needed for exact OptionMetrics lags is written to a separate
internal file and can never enter the 59-row signal-date publication scope.

No function in this module reads labels, forward returns, portfolio weights,
trades, P&L, or final-period data.  The universe context uses only return
availability/usability flags, never a return value.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
import os
import shutil
import tempfile
import threading
import tomllib
from contextlib import contextmanager
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Iterator, Mapping, Sequence

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from portfolio_bot.data import phase_02_auxiliary_compustat_factors as compustat
from portfolio_bot.data import phase_02_optionmetrics as optionmetrics
from portfolio_bot.data import phase_02_sec_restatement_auxiliary as sec
from portfolio_bot.data import phase_02_signal_date_universe as frozen_universe
from portfolio_bot.data.phase_02_auxiliary_core import (
    TradingCalendar as AuxiliaryTradingCalendar,
    load_schema_document,
    sha256 as auxiliary_sha256,
    table_from_rows,
)
from portfolio_bot.data.phase_02_auxiliary_linkage import (
    build_auxiliary_signal_linkage,
)
from portfolio_bot.data.phase_02_crsp_typed_backbone_core import (
    load_schema_registry,
)
from portfolio_bot.data.phase_02_optionmetrics import (
    OptionMetricsSettings,
    build_crsp_links,
    build_name_intervals,
    stream_historical_volatility,
    stream_option_volume,
)
from portfolio_bot.data.phase_02_sec_restatement_auxiliary import (
    build_sec_restatement_family,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
POLICY_PATH = (
    PROJECT_ROOT
    / "configs/phase_05_validation/stage_11_universe_auxiliary_extension_v1.toml"
)
UNIVERSE_SCHEMA_PATH = (
    PROJECT_ROOT
    / "configs/phase_02_canonicalization/signal_date_universe_schema_v1.json"
)
AUXILIARY_SCHEMA_PATH = (
    PROJECT_ROOT
    / "configs/phase_02_canonicalization/auxiliary_source_schema_v1.json"
)
DEVELOPMENT_AUXILIARY_PRIMARY = (
    PROJECT_ROOT
    / "artifacts/phase_02_canonical_panel/LATEST_AUXILIARY_AVAILABILITY_LINKAGE_PRIMARY.json"
)
DEVELOPMENT_AUXILIARY_VERIFIED = (
    PROJECT_ROOT
    / "artifacts/phase_02_canonical_panel/LATEST_AUXILIARY_AVAILABILITY_LINKAGE_VERIFIED.json"
)
PATHS_PATH = PROJECT_ROOT / "configs/paths.toml"

POLICY_ID = "phase05-stage11-universe-auxiliary-extension-v1"
CONTRACT_ID = "phase05-stage11-universe-auxiliary-extension-v1"

DEVELOPMENT_CONTEXT_START = dt.date(2000, 1, 1)
AUXILIARY_EXTENSION_START = dt.date(2014, 11, 1)
INTERNAL_CONTEXT_SIGNAL = dt.date(2014, 11, 28)
FIRST_SCORED_SIGNAL = dt.date(2014, 12, 31)
MARKET_EXTENSION_START = dt.date(2015, 1, 1)
LAST_SCORED_SIGNAL = dt.date(2019, 10, 31)
FIRST_PROHIBITED_SIGNAL = dt.date(2019, 11, 29)
LAST_PERMITTED_EXIT = dt.date(2019, 12, 2)
FIRST_FINAL_ENTRY = dt.date(2020, 1, 2)
REQUIRED_SCORED_SIGNALS = 59

FACTOR_DATE_COLUMNS = MappingProxyType(
    {
        "ff_factor_daily_observation.parquet": "observation_date",
        "ff_factor_monthly_observation.parquet": "factor_month",
        "ps_liquidity_observation.parquet": "observation_month",
    }
)

# These fixed limits affect only DuckDB execution strategy.  They do not alter
# any row scope, ordering, schema, or formula, and are recorded in the receipt.
# The exact multiset overlap gates below can exceed 8 GB on the full CRSP
# history.  Fewer workers reduce per-thread buffering while a 16 GB ceiling
# leaves enough room for DuckDB's exact EXCEPT ALL operation to complete.
FULL_CONTEXT_DUCKDB_THREADS = 4
FULL_CONTEXT_DUCKDB_MEMORY_LIMIT = "16GB"

SCORED_UNIVERSE_FILE = "validation_signal_date_universe.parquet"
CONTEXT_UNIVERSE_FILE = "internal_context_signal_date_universe.parquet"
SCORED_LINKAGE_FILE = "auxiliary_signal_linkage.parquet"
CONTEXT_LINKAGE_FILE = "internal_context_auxiliary_signal_linkage.parquet"
RECEIPT_FILE = "universe_auxiliary_receipt.json"

_UNIVERSE_SCOPE_LOCK = threading.Lock()
_AUXILIARY_SCOPE_LOCK = threading.Lock()

Progress = Callable[[str, Mapping[str, Any]], None]


class ValidationUniverseAuxiliaryError(RuntimeError):
    """Raised when a validation scope or point-in-time gate fails closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationUniverseAuxiliaryError(message)


def _canonical_json(value: Any) -> bytes:
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


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + f".partial-{os.getpid()}")
    temporary.write_bytes(_canonical_json(value))
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def _json_object(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text("utf-8"))
    _require(isinstance(value, dict), f"invalid {label} JSON")
    return value


def _authenticate_development_auxiliary_root(path: Path) -> dict[str, str]:
    primary = _json_object(DEVELOPMENT_AUXILIARY_PRIMARY, "development auxiliary primary")
    verified = _json_object(DEVELOPMENT_AUXILIARY_VERIFIED, "development auxiliary verified")
    _require(primary.get("status") == "auxiliary_primary_passed_pending_independent_verification", "wrong development auxiliary primary status")
    _require(verified.get("status") == "verified_passed", "development auxiliary is not verified")
    _require(
        primary.get("build_id") == verified.get("build_id")
        and primary.get("manifest_sha256") == verified.get("primary_manifest_sha256"),
        "development auxiliary primary/verification identities differ",
    )
    with PATHS_PATH.open("rb") as handle:
        database_root = Path(tomllib.load(handle)["database"]["root"]).resolve()
    expected = (database_root / str(primary["interim_path"])).resolve()
    _require(path.resolve() == expected, "development auxiliary root is not the verified build")
    manifest = PROJECT_ROOT / str(primary["artifact_path"]) / "manifest.json"
    _require(
        manifest.is_file() and _sha256(manifest) == primary["manifest_sha256"],
        "development auxiliary manifest differs",
    )
    return {
        "build_id": str(primary["build_id"]),
        "verification_id": str(verified["verification_id"]),
        "manifest_sha256": str(primary["manifest_sha256"]),
    }


def _emit(progress: Progress | None, event: str, **details: Any) -> None:
    if progress is not None:
        progress(event, MappingProxyType(details))


@dataclasses.dataclass(frozen=True, slots=True)
class ValidationUniverseAuxiliaryScope:
    """The only permitted Stage 11 universe/auxiliary date scope."""

    context_signal: dt.date = INTERNAL_CONTEXT_SIGNAL
    first_scored_signal: dt.date = FIRST_SCORED_SIGNAL
    last_scored_signal: dt.date = LAST_SCORED_SIGNAL
    last_exit: dt.date = LAST_PERMITTED_EXIT

    def __post_init__(self) -> None:
        observed = (
            self.context_signal,
            self.first_scored_signal,
            self.last_scored_signal,
            self.last_exit,
        )
        _require(
            all(type(value) is dt.date for value in observed),
            "validation scope values must be exact dates",
        )
        _require(
            observed
            == (
                INTERNAL_CONTEXT_SIGNAL,
                FIRST_SCORED_SIGNAL,
                LAST_SCORED_SIGNAL,
                LAST_PERMITTED_EXIT,
            ),
            "Stage 11 universe/auxiliary scope may not be widened or shifted",
        )
        _require(
            self.context_signal < self.first_scored_signal
            <= self.last_scored_signal < self.last_exit < FIRST_FINAL_ENTRY,
            "Stage 11 universe/auxiliary scope is not ordered",
        )


DEFAULT_SCOPE = ValidationUniverseAuxiliaryScope()


@dataclasses.dataclass(frozen=True, slots=True)
class ValidationUniverseInputs:
    """Precomposed, predictor-only inputs consumed by the frozen formula."""

    typed_root: Path
    return_root: Path
    calendar_path: Path

    def frozen_settings(self) -> frozen_universe.Settings:
        return frozen_universe.Settings(
            database_root=self.typed_root.parent,
            typed_root=self.typed_root,
            return_root=self.return_root,
            calendar_path=self.calendar_path,
            phase_config={},
        )


@dataclasses.dataclass(frozen=True, slots=True)
class FullPredictorContextInputs:
    """Frozen development inputs plus isolated Stage 11 extensions."""

    development_typed_root: Path
    development_return_root: Path
    development_auxiliary_root: Path
    validation_market_root: Path
    validation_return_root: Path
    validation_auxiliary_root: Path
    calendar_path: Path


@dataclasses.dataclass(frozen=True, slots=True)
class ScopedUniverseResult:
    scored: frozen_universe.UniverseResult
    internal_context: frozen_universe.UniverseResult
    schedule: Mapping[str, Any]


def load_policy(path: Path = POLICY_PATH) -> Mapping[str, Any]:
    with path.open("rb") as handle:
        value = tomllib.load(handle)
    _require(value.get("policy_id") == POLICY_ID, "wrong universe/auxiliary policy ID")
    scope = value.get("scope", {})
    expected = {
        "internal_context_signal_date": INTERNAL_CONTEXT_SIGNAL.isoformat(),
        "first_scored_signal_date": FIRST_SCORED_SIGNAL.isoformat(),
        "last_scored_signal_date": LAST_SCORED_SIGNAL.isoformat(),
        "scored_signal_count": REQUIRED_SCORED_SIGNALS,
        "last_permitted_exit_date": LAST_PERMITTED_EXIT.isoformat(),
        "first_prohibited_signal_date": FIRST_PROHIBITED_SIGNAL.isoformat(),
        "first_final_entry_date": FIRST_FINAL_ENTRY.isoformat(),
    }
    _require(scope == expected, "universe/auxiliary policy scope differs")
    boundaries = value.get("boundaries", {})
    _require(
        boundaries.get("labels_or_forward_returns_permitted") is False
        and boundaries.get("portfolio_weights_trades_or_pnl_permitted") is False
        and boundaries.get("final_2020_2025_rows_permitted") is False
        and boundaries.get("frozen_stage02_source_modification_permitted") is False,
        "universe/auxiliary policy does not fail closed",
    )
    _require(
        value.get("auxiliary", {}).get("segment_feature_support")
        == "frozen_inactive"
        and value["auxiliary"].get("segment_observation_rows") == 0,
        "segment features must remain frozen inactive",
    )
    auxiliary = value.get("auxiliary", {})
    _require(
        auxiliary.get("optionmetrics_scope_start")
        == AUXILIARY_EXTENSION_START.isoformat()
        and auxiliary.get("optionmetrics_scope_end")
        == LAST_SCORED_SIGNAL.isoformat()
        and auxiliary.get("compustat_last_public_availability_date")
        == LAST_SCORED_SIGNAL.isoformat()
        and auxiliary.get("factor_last_observation_date")
        == LAST_SCORED_SIGNAL.isoformat()
        and auxiliary.get("sec_last_public_availability_date")
        == LAST_SCORED_SIGNAL.isoformat(),
        "auxiliary extension scope differs",
    )
    return MappingProxyType(value)


def _calendar_arrays(
    path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    table = pq.read_table(
        path,
        columns=["calendar_date", "is_observed_month_end"],
        use_threads=False,
    )
    dates = (
        table["calendar_date"]
        .combine_chunks()
        .to_numpy(zero_copy_only=False)
        .astype("datetime64[D]")
        .astype(np.int64)
    )
    month_end_array = table["is_observed_month_end"].combine_chunks()
    _require(month_end_array.null_count == 0, "calendar month-end flag contains nulls")
    month_end = month_end_array.to_numpy(zero_copy_only=False).astype(bool)
    _require(len(dates) > 0, "trading calendar is empty")
    _require(np.all(dates[1:] > dates[:-1]), "trading calendar is not sorted unique")
    return dates, month_end


def _validation_signal_calendar(
    settings: frozen_universe.Settings,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Frozen-calendar schedule with one internal and 59 scored signals."""

    calendar, month_end = _calendar_arrays(settings.calendar_path)
    month_end_positions = np.flatnonzero(month_end)
    month_end_dates = calendar[month_end_positions]
    context = np.datetime64(INTERNAL_CONTEXT_SIGNAL, "D").astype(np.int64)
    first = np.datetime64(FIRST_SCORED_SIGNAL, "D").astype(np.int64)
    last = np.datetime64(LAST_SCORED_SIGNAL, "D").astype(np.int64)
    selected_positions = month_end_positions[
        (month_end_dates == context)
        | ((month_end_dates >= first) & (month_end_dates <= last))
    ]
    selected = calendar[selected_positions]
    _require(len(selected) == REQUIRED_SCORED_SIGNALS + 1, "calendar is not 1+59 signals")
    _require(
        int(selected[0]) == int(context)
        and int(selected[1]) == int(first)
        and int(selected[-1]) == int(last),
        "calendar validation signal boundaries differ",
    )
    next_month_end_indices = np.searchsorted(
        month_end_positions, selected_positions, side="right"
    )
    _require(
        np.all(next_month_end_indices < len(month_end_positions)),
        "calendar lacks a later month end for validation",
    )
    entry_positions = selected_positions + 1
    next_month_end_positions = month_end_positions[next_month_end_indices]
    next_month_end_dates = calendar[next_month_end_positions]
    terminal_exit = np.datetime64(LAST_PERMITTED_EXIT, "D").astype(np.int64)
    # The Stage 13 terminal December interval is deliberately partial and may
    # end on the authorized month-end itself.  Ordinary intervals retain the
    # frozen next-trading-day exit rule.
    exit_positions = next_month_end_positions + (
        next_month_end_dates != terminal_exit
    ).astype(np.int64)
    _require(
        np.all(entry_positions < len(calendar)) and np.all(exit_positions < len(calendar)),
        "calendar lacks a required validation entry or exit",
    )
    _require(
        np.all(calendar[entry_positions] > selected)
        and np.all(
            (calendar[exit_positions] > next_month_end_dates)
            | (
                (next_month_end_dates == terminal_exit)
                & (calendar[exit_positions] == terminal_exit)
            )
        ),
        "validation schedule is not strictly forward",
    )
    _require(
        int(calendar[exit_positions[-1]])
        == int(np.datetime64(LAST_PERMITTED_EXIT, "D").astype(np.int64)),
        "last validation exit differs from 2019-12-02",
    )
    return calendar, selected, calendar[entry_positions], calendar[exit_positions]


@contextmanager
def _frozen_universe_scope() -> Iterator[None]:
    """Install the exact calendar adapter for one process-local computation."""

    acquired = _UNIVERSE_SCOPE_LOCK.acquire(blocking=False)
    _require(acquired, "another validation-universe scope is already active")
    original = frozen_universe._signal_calendar
    frozen_universe._signal_calendar = _validation_signal_calendar
    try:
        yield
    finally:
        frozen_universe._signal_calendar = original
        _UNIVERSE_SCOPE_LOCK.release()


def _subset_columns(
    columns: Mapping[str, Any],
    indices: np.ndarray,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, values in columns.items():
        if isinstance(values, np.ndarray):
            result[name] = values[indices]
        else:
            result[name] = [values[int(index)] for index in indices]
    return result


def _summary(columns: Mapping[str, Any], *, expected_signal_dates: int) -> dict[str, Any]:
    signals = list(columns["signal_date"])
    _require(bool(signals), "universe subset is empty")
    unique_signals = sorted(set(signals))
    _require(
        len(unique_signals) == expected_signal_dates,
        "universe subset does not contain every required signal date",
    )
    long_eligible = np.asarray(columns["long_eligible"], dtype=bool)
    short_eligible = np.asarray(columns["short_eligible"], dtype=bool)
    base_eligible = np.asarray(columns["base_eligible"], dtype=bool)
    selected = np.asarray(columns["issuer_line_selected"], dtype=bool)
    long_pre = np.asarray(columns["long_eligible_pre_dedup"], dtype=bool)
    short_pre = np.asarray(columns["short_eligible_pre_dedup"], dtype=bool)
    us_flag = np.asarray(columns["us_incorporation_flag"], dtype=str)
    share_type = np.asarray(columns["share_type"], dtype=str)
    by_signal_long = [
        int(sum(day == value and long_eligible[index] for index, day in enumerate(signals)))
        for value in unique_signals
    ]
    by_signal_short = [
        int(sum(day == value and short_eligible[index] for index, day in enumerate(signals)))
        for value in unique_signals
    ]
    return {
        "signal_dates": len(unique_signals),
        "first_signal_date": unique_signals[0].isoformat(),
        "last_signal_date": unique_signals[-1].isoformat(),
        "rows": len(signals),
        "identity_matched_rows": len(signals),
        "base_screen_rows": int(base_eligible.sum()),
        "long_pre_dedup_rows": int(long_pre.sum()),
        "issuer_primary_rows": int(selected.sum()),
        "long_eligible_rows": int(long_eligible.sum()),
        "short_pre_dedup_rows": int(short_pre.sum()),
        "short_eligible_rows": int(short_eligible.sum()),
        "foreign_long_eligible_rows": int(np.sum(long_eligible & (us_flag != "Y"))),
        "adr_long_eligible_rows": int(np.sum(long_eligible & (share_type == "AD"))),
        "signal_long_count_min": min(by_signal_long),
        "signal_long_count_max": max(by_signal_long),
        "signal_short_count_min": min(by_signal_short),
        "signal_short_count_max": max(by_signal_short),
        "point_in_time_violations": 0,
        "future_outcome_filter_applied": False,
    }


def compute_validation_universe(
    inputs: ValidationUniverseInputs,
    *,
    scope: ValidationUniverseAuxiliaryScope = DEFAULT_SCOPE,
) -> ScopedUniverseResult:
    """Run the frozen universe code, then isolate context from scored rows."""

    _require(scope == DEFAULT_SCOPE, "only the frozen Stage 11 scope is supported")
    load_policy()
    for path in (
        inputs.typed_root / "crsp_security_day.parquet",
        inputs.typed_root / "crsp_identity_interval.parquet",
        inputs.return_root / "crsp_security_day_return.parquet",
        inputs.calendar_path,
    ):
        _require(path.is_file() and not path.is_symlink(), f"missing or unsafe input: {path}")
    with _frozen_universe_scope():
        full = frozen_universe.compute_universe(inputs.frozen_settings())

    dates = np.asarray(full.columns["signal_date"], dtype="datetime64[D]")
    context_mask = dates == np.datetime64(INTERNAL_CONTEXT_SIGNAL, "D")
    scored_mask = (dates >= np.datetime64(FIRST_SCORED_SIGNAL, "D")) & (
        dates <= np.datetime64(LAST_SCORED_SIGNAL, "D")
    )
    _require(np.all(context_mask | scored_mask), "universe contains an out-of-scope signal")
    context_indices = np.flatnonzero(context_mask)
    scored_indices = np.flatnonzero(scored_mask)
    context_columns = _subset_columns(full.columns, context_indices)
    scored_columns = _subset_columns(full.columns, scored_indices)
    context_summary = _summary(context_columns, expected_signal_dates=1)
    scored_summary = _summary(
        scored_columns, expected_signal_dates=REQUIRED_SCORED_SIGNALS
    )
    _require(
        scored_summary["first_signal_date"] == FIRST_SCORED_SIGNAL.isoformat()
        and scored_summary["last_signal_date"] == LAST_SCORED_SIGNAL.isoformat(),
        "scored universe boundaries differ",
    )
    _require(
        context_summary["first_signal_date"] == INTERNAL_CONTEXT_SIGNAL.isoformat(),
        "internal context signal differs",
    )
    schedule = {
        "internal_context_signal": INTERNAL_CONTEXT_SIGNAL.isoformat(),
        "first_scored_signal": FIRST_SCORED_SIGNAL.isoformat(),
        "last_scored_signal": LAST_SCORED_SIGNAL.isoformat(),
        "scored_signal_count": REQUIRED_SCORED_SIGNALS,
        "last_permitted_exit": LAST_PERMITTED_EXIT.isoformat(),
        "context_signal_rows_published": 0,
    }
    return ScopedUniverseResult(
        scored=frozen_universe.UniverseResult(scored_columns, scored_summary),
        internal_context=frozen_universe.UniverseResult(
            context_columns, context_summary
        ),
        schedule=MappingProxyType(schedule),
    )


def validation_universe_schema() -> pa.Schema:
    registry = load_schema_registry(UNIVERSE_SCHEMA_PATH)
    frozen = registry.tables["crsp_signal_date_universe"].arrow_schema()
    metadata = dict(frozen.metadata or {})
    metadata.update(
        {
            b"contract_id": CONTRACT_ID.encode("ascii"),
            b"role": b"stage11_point_in_time_signal_universe",
            b"scope_first_scored_signal": FIRST_SCORED_SIGNAL.isoformat().encode("ascii"),
            b"scope_last_scored_signal": LAST_SCORED_SIGNAL.isoformat().encode("ascii"),
        }
    )
    return frozen.with_metadata(metadata)


def _validate_universe_table(
    path: Path,
    *,
    expected_signal_dates: int,
    first_signal: dt.date,
    last_signal: dt.date,
) -> dict[str, Any]:
    file = pq.ParquetFile(path)
    schema = validation_universe_schema()
    _require(
        file.schema_arrow.equals(schema, check_metadata=True),
        f"universe schema differs: {path.name}",
    )
    table = pq.read_table(
        path,
        columns=[
            "signal_date",
            "permno",
            "observation_date",
            "public_availability_date",
            "availability_not_after_signal",
            "return_observations_after_signal_count",
            "future_outcome_filter_applied",
            "share_type",
            "primary_exchange",
            "long_eligible",
            "short_eligible",
        ],
        use_threads=False,
    )
    rows = table.to_pylist()
    _require(bool(rows), f"universe table is empty: {path.name}")
    keys = [(row["signal_date"], int(row["permno"])) for row in rows]
    _require(keys == sorted(set(keys)), f"universe key is not sorted unique: {path.name}")
    signals = sorted({key[0] for key in keys})
    _require(
        len(signals) == expected_signal_dates
        and signals[0] == first_signal
        and signals[-1] == last_signal,
        f"universe signal scope differs: {path.name}",
    )
    for row in rows:
        _require(
            row["observation_date"] <= row["signal_date"]
            and row["public_availability_date"] <= row["signal_date"]
            and row["availability_not_after_signal"] is True
            and int(row["return_observations_after_signal_count"]) == 0
            and row["future_outcome_filter_applied"] is False,
            f"point-in-time universe gate failed: {path.name}",
        )
        _require(
            row["share_type"] in {"NS", "AD"}
            or row["long_eligible"] is False,
            "an ineligible share type entered the long universe",
        )
        _require(
            row["primary_exchange"] in {"N", "A", "Q"}
            or row["long_eligible"] is False,
            "an ineligible exchange entered the long universe",
        )
        _require(
            row["short_eligible"] is False or row["long_eligible"] is True,
            "short eligibility does not imply long eligibility",
        )
    return {
        "path": path.name,
        "rows": len(rows),
        "row_groups": file.num_row_groups,
        "signal_dates": len(signals),
        "minimum_signal_date": signals[0].isoformat(),
        "maximum_signal_date": signals[-1].isoformat(),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def write_validation_universe(
    result: ScopedUniverseResult,
    output_directory: Path,
) -> dict[str, Any]:
    """Write scored and internal context universes to disjoint files."""

    output = output_directory.resolve()
    if output.exists():
        _require(output.is_dir() and not any(output.iterdir()), "universe output is not empty")
    else:
        output.mkdir(parents=True)
    schema = validation_universe_schema()
    scored_path = output / SCORED_UNIVERSE_FILE
    context_path = output / CONTEXT_UNIVERSE_FILE
    pq.write_table(
        pa.Table.from_pydict(result.scored.columns, schema=schema),
        scored_path,
        compression="zstd",
        use_dictionary=True,
        write_statistics=True,
        row_group_size=131_072,
    )
    pq.write_table(
        pa.Table.from_pydict(result.internal_context.columns, schema=schema),
        context_path,
        compression="zstd",
        use_dictionary=True,
        write_statistics=True,
        row_group_size=131_072,
    )
    receipt = {
        "schema_version": 1,
        "status": "stage11_validation_universe_built_no_outcome_access",
        "contract_id": CONTRACT_ID,
        "scope": dict(result.schedule),
        "scored": _validate_universe_table(
            scored_path,
            expected_signal_dates=REQUIRED_SCORED_SIGNALS,
            first_signal=FIRST_SCORED_SIGNAL,
            last_signal=LAST_SCORED_SIGNAL,
        ),
        "internal_context": _validate_universe_table(
            context_path,
            expected_signal_dates=1,
            first_signal=INTERNAL_CONTEXT_SIGNAL,
            last_signal=INTERNAL_CONTEXT_SIGNAL,
        ),
        "frozen_formula_binding": {
            "module": "src/portfolio_bot/data/phase_02_signal_date_universe.py",
            "sha256": _sha256(Path(frozen_universe.__file__)),
            "source_modified": False,
        },
        "authorizations": {
            "validation_predictor_construction_authorized": False,
            "labels_forward_returns_weights_trades_pnl_authorized": False,
            "final_2020_2025_access_authorized": False,
        },
    }
    _atomic_json(output / RECEIPT_FILE, receipt)
    return receipt


_MARKET_CONTEXT_COLUMNS = (
    "permno",
    "calendar_date",
    "conditional_type",
    "close_usd_per_share",
    "market_cap_usd",
    "volume_shares",
    "screening_dollar_volume_usd",
    "screening_price_valid",
    "market_cap_valid",
    "liquidity_session_valid",
)
_RETURN_CONTEXT_COLUMNS = (
    "permno",
    "calendar_date",
    "outcome_availability_date",
    "return_history_usable",
)


def _sql_path(path: Path) -> str:
    return str(path.resolve()).replace("'", "''")


def _required_parquet_columns(
    path: Path,
    columns: Sequence[str],
    label: str,
) -> None:
    _require(path.is_file() and not path.is_symlink(), f"missing or unsafe {label}")
    names = set(pq.ParquetFile(path).schema_arrow.names)
    missing = sorted(set(columns) - names)
    _require(not missing, f"{label} lacks columns {missing}")


def _history_start(calendar_path: Path) -> dt.date:
    dates, _ = _calendar_arrays(calendar_path)
    context = np.datetime64(INTERNAL_CONTEXT_SIGNAL, "D").astype(np.int64)
    position = int(np.searchsorted(dates, context))
    _require(position < len(dates) and int(dates[position]) == int(context), "context signal absent")
    _require(position >= 251, "calendar lacks 252-session return context")
    ordinal = int(dates[position - 251])
    return dt.date(1970, 1, 1) + dt.timedelta(days=ordinal)


def _copy_ordered_context(
    connection: duckdb.DuckDBPyConnection,
    *,
    development_path: Path,
    extension_path: Path,
    output_path: Path,
    columns: Sequence[str],
    history_start: dt.date,
) -> None:
    projection = ",".join(f'"{name}"' for name in columns)
    start = history_start.isoformat()
    end = LAST_SCORED_SIGNAL.isoformat()
    query = f"""
        COPY (
            SELECT {projection}
            FROM (
                SELECT {projection}
                FROM read_parquet('{_sql_path(development_path)}')
                WHERE calendar_date BETWEEN DATE '{start}' AND DATE '{end}'
                UNION ALL
                SELECT {projection}
                FROM read_parquet('{_sql_path(extension_path)}')
                WHERE calendar_date BETWEEN DATE '{start}' AND DATE '{end}'
            ) source
            ORDER BY permno, calendar_date
        ) TO '{_sql_path(output_path)}'
        (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 131072)
    """
    connection.execute(query)


def compose_validation_universe_inputs(
    *,
    development_typed_root: Path,
    development_return_root: Path,
    validation_market_root: Path,
    validation_return_root: Path,
    calendar_path: Path,
    output_root: Path,
    progress: Progress | None = None,
) -> tuple[ValidationUniverseInputs, dict[str, Any]]:
    """Compose only the bounded predictor context needed by universe screens.

    Development and validation files remain immutable.  The composed market
    table carries just the frozen screening columns; the composed return table
    carries only dates and usability/availability flags.  Numeric returns are
    therefore structurally unavailable to this operation.
    """

    load_policy()
    market_development = development_typed_root / "crsp_security_day.parquet"
    market_extension = validation_market_root / "crsp_security_day.parquet"
    return_development = development_return_root / "crsp_security_day_return.parquet"
    return_extension = validation_return_root / "crsp_security_day_return.parquet"
    identity_extension = validation_market_root / "crsp_identity_interval.parquet"
    for path, columns, label in (
        (market_development, _MARKET_CONTEXT_COLUMNS, "development market"),
        (market_extension, _MARKET_CONTEXT_COLUMNS, "validation market extension"),
        (return_development, _RETURN_CONTEXT_COLUMNS, "development return audit"),
        (return_extension, _RETURN_CONTEXT_COLUMNS, "validation return audit extension"),
        (identity_extension, ("permno", "start_date", "end_date"), "validation identity"),
    ):
        _required_parquet_columns(path, columns, label)
    _require(calendar_path.is_file() and not calendar_path.is_symlink(), "unsafe calendar")

    output = output_root.resolve()
    if output.exists():
        _require(output.is_dir() and not any(output.iterdir()), "context output is not empty")
    else:
        output.mkdir(parents=True)
    typed_root = output / "typed"
    return_root = output / "returns"
    typed_root.mkdir()
    return_root.mkdir()
    history_start = _history_start(calendar_path)
    _emit(progress, "validation_universe_context_started", history_start=history_start.isoformat())

    spill = output / ".duckdb_spill"
    spill.mkdir()
    connection = duckdb.connect(":memory:")
    try:
        connection.execute("SET preserve_insertion_order = false")
        connection.execute("SET threads = 2")
        connection.execute("SET memory_limit = '2GB'")
        connection.execute("SET temp_directory = ?", [str(spill)])
        for path, label in (
            (market_extension, "validation market extension"),
            (return_extension, "validation return audit extension"),
        ):
            minimum, maximum = connection.execute(
                f"SELECT min(calendar_date), max(calendar_date) FROM read_parquet('{_sql_path(path)}')"
            ).fetchone()
            _require(minimum is not None and maximum is not None, f"empty {label}")
            _require(minimum >= MARKET_EXTENSION_START, f"{label} overlaps development")
            _require(maximum <= LAST_PERMITTED_EXIT, f"{label} exceeds validation exit")
        _copy_ordered_context(
            connection,
            development_path=market_development,
            extension_path=market_extension,
            output_path=typed_root / "crsp_security_day.parquet",
            columns=_MARKET_CONTEXT_COLUMNS,
            history_start=history_start,
        )
        _copy_ordered_context(
            connection,
            development_path=return_development,
            extension_path=return_extension,
            output_path=return_root / "crsp_security_day_return.parquet",
            columns=_RETURN_CONTEXT_COLUMNS,
            history_start=history_start,
        )
        connection.read_parquet(str(identity_extension)).write_parquet(
            str(typed_root / "crsp_identity_interval.parquet"),
            compression="zstd",
        )
        connection.read_parquet(
            str(typed_root / "crsp_security_day.parquet")
        ).create_view("market_context")
        connection.read_parquet(
            str(return_root / "crsp_security_day_return.parquet")
        ).create_view("return_context")
        market_rows = int(connection.execute("SELECT count(*) FROM market_context").fetchone()[0])
        return_rows = int(connection.execute("SELECT count(*) FROM return_context").fetchone()[0])
        duplicate_market = int(
            connection.execute(
                """
                SELECT count(*) FROM (
                    SELECT permno, calendar_date
                    FROM market_context
                    GROUP BY permno, calendar_date
                    HAVING count(*) <> 1
                )
                """
            ).fetchone()[0]
        )
        duplicate_return = int(
            connection.execute(
                """
                SELECT count(*) FROM (
                    SELECT permno, calendar_date
                    FROM return_context
                    GROUP BY permno, calendar_date
                    HAVING count(*) <> 1
                )
                """
            ).fetchone()[0]
        )
        key_differences = int(
            connection.execute(
                """
                SELECT count(*)
                FROM market_context m
                FULL OUTER JOIN return_context r USING (permno, calendar_date)
                WHERE m.permno IS NULL OR r.permno IS NULL
                """
            ).fetchone()[0]
        )
    finally:
        connection.close()
        try:
            spill.rmdir()
        except OSError:
            pass
    _require(market_rows == return_rows and market_rows > 0, "context row counts differ")
    _require(duplicate_market == duplicate_return == 0, "context keys are duplicated")
    _require(key_differences == 0, "market and return-audit context keys differ")

    outputs = {}
    for role, path in (
        ("market", typed_root / "crsp_security_day.parquet"),
        ("identity", typed_root / "crsp_identity_interval.parquet"),
        ("return_audit", return_root / "crsp_security_day_return.parquet"),
    ):
        file = pq.ParquetFile(path)
        outputs[role] = {
            "path": str(path.relative_to(output)),
            "rows": file.metadata.num_rows,
            "row_groups": file.num_row_groups,
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
    receipt = {
        "schema_version": 1,
        "status": "stage11_validation_universe_context_composed",
        "history_start": history_start.isoformat(),
        "last_context_date": LAST_SCORED_SIGNAL.isoformat(),
        "rows": market_rows,
        "duplicate_market_keys": duplicate_market,
        "duplicate_return_audit_keys": duplicate_return,
        "market_return_key_differences": key_differences,
        "numeric_return_columns_read": 0,
        "labels_forward_returns_or_final_rows_read": 0,
        "outputs": outputs,
    }
    _atomic_json(output / "context_receipt.json", receipt)
    _emit(progress, "validation_universe_context_complete", rows=market_rows)
    return ValidationUniverseInputs(typed_root, return_root, calendar_path), receipt


def _field_schema(path: Path) -> pa.Schema:
    return pq.ParquetFile(path).schema_arrow.remove_metadata()


def _logical_field_signature(path: Path) -> tuple[tuple[str, str], ...]:
    return tuple(
        (field.name, str(field.type))
        for field in pq.ParquetFile(path).schema_arrow
    )


def _require_same_fields(left: Path, right: Path, label: str) -> None:
    _require(
        _logical_field_signature(left) == _logical_field_signature(right),
        f"development and validation field schemas differ for {label}",
    )


def _copy_query(
    connection: duckdb.DuckDBPyConnection,
    query: str,
    destination: Path,
) -> None:
    connection.execute(
        f"""
        COPY ({query}) TO '{_sql_path(destination)}'
        (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 131072)
        """
    )


def _table_difference_count(
    connection: duckdb.DuckDBPyConnection,
    left_query: str,
    right_query: str,
) -> int:
    # A monolithic EXCEPT ALL over the full CRSP history can retain almost the
    # entire multiset in memory.  Partition both inputs by DuckDB's deterministic
    # row hash, then perform the same exact comparison inside bounded buckets.
    # Hash collisions cannot create false equality: complete rows are still
    # compared by EXCEPT ALL after they land in the same bucket.
    described = connection.execute(
        f"DESCRIBE SELECT * FROM ({left_query})"
    ).fetchall()
    columns = tuple(str(row[0]) for row in described)
    _require(bool(columns), "cannot compare tables without fields")
    right_columns = tuple(
        str(row[0])
        for row in connection.execute(
            f"DESCRIBE SELECT * FROM ({right_query})"
        ).fetchall()
    )
    _require(columns == right_columns, "table comparison field order differs")
    partition_column = "__stage13_exact_partition"
    _require(partition_column not in columns, "reserved comparison field exists")
    quoted = ",".join('"' + name.replace('"', '""') + '"' for name in columns)
    partition_count = 128
    temp_root = Path(
        str(connection.execute("SELECT current_setting('temp_directory')").fetchone()[0])
    )
    temp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="exact-table-compare-", dir=temp_root
    ) as temporary:
        comparison_root = Path(temporary)
        left_root = comparison_root / "left"
        right_root = comparison_root / "right"
        for query, destination in (
            (left_query, left_root),
            (right_query, right_root),
        ):
            connection.execute(
                f"""
                COPY (
                    SELECT *,
                           hash({quoted}) % {partition_count}
                               AS {partition_column}
                    FROM ({query})
                ) TO '{_sql_path(destination)}'
                (FORMAT PARQUET, COMPRESSION ZSTD,
                 PARTITION_BY ({partition_column}), ROW_GROUP_SIZE 131072)
                """
            )

        bucket_names = {
            path.name
            for root in (left_root, right_root)
            if root.is_dir()
            for path in root.iterdir()
            if path.is_dir()
        }
        differences = 0
        empty = f"SELECT {quoted} FROM ({left_query}) WHERE false"
        for bucket_name in sorted(bucket_names):
            left_bucket = left_root / bucket_name / "*.parquet"
            right_bucket = right_root / bucket_name / "*.parquet"
            left_bucket_query = (
                f"SELECT {quoted} FROM read_parquet("
                f"'{_sql_path(left_bucket)}', hive_partitioning=false)"
                if (left_root / bucket_name).is_dir()
                else empty
            )
            right_bucket_query = (
                f"SELECT {quoted} FROM read_parquet("
                f"'{_sql_path(right_bucket)}', hive_partitioning=false)"
                if (right_root / bucket_name).is_dir()
                else empty
            )
            differences += int(
                connection.execute(
                    f"SELECT count(*) FROM (({left_bucket_query}) "
                    f"EXCEPT ALL ({right_bucket_query}))"
                ).fetchone()[0]
            )
            differences += int(
                connection.execute(
                    f"SELECT count(*) FROM (({right_bucket_query}) "
                    f"EXCEPT ALL ({left_bucket_query}))"
                ).fetchone()[0]
            )
        return differences


def _key_gate(
    connection: duckdb.DuckDBPyConnection,
    path: Path,
    keys: Sequence[str],
) -> tuple[int, int]:
    key_sql = ",".join(f'"{key}"' for key in keys)
    distinct_rows = int(connection.execute(
        f"""
        SELECT count(*)
        FROM (
            SELECT {key_sql}
            FROM read_parquet('{_sql_path(path)}')
            GROUP BY {key_sql}
        )
        """
    ).fetchone()[0])
    total = int(
        connection.execute(
            f"SELECT count(*) FROM read_parquet('{_sql_path(path)}')"
        ).fetchone()[0]
    )
    return total, total - distinct_rows


def _sorted_unique_scan(path: Path, keys: Sequence[str]) -> None:
    previous: tuple[Any, ...] | None = None
    for batch in pq.ParquetFile(path).iter_batches(
        batch_size=131_072,
        columns=list(keys),
        use_threads=False,
    ):
        columns = [batch.column(index).to_pylist() for index in range(len(keys))]
        for key in zip(*columns, strict=True):
            _require(
                previous is None or key > previous,
                f"predictor context is not sorted unique: {path.name}",
            )
            previous = key


def compose_full_validation_predictor_context(
    inputs: FullPredictorContextInputs,
    output_root: Path,
    *,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Compose the exact 2000--2019 history required by Stage 6 formulas.

    Historical realized returns through each signal are legitimate lagged
    predictor/risk inputs and are retained.  Both CRSP return columns
    ``raw_return_ex_distributions`` and ``authoritative_total_return`` are
    explicitly required.  The branch ends on 2019-10-31, so no later outcome
    can be smuggled into predictor materialization.
    """

    load_policy()
    output = output_root.resolve()
    if output.exists():
        _require(output.is_dir() and not any(output.iterdir()), "predictor context output is not empty")
    else:
        output.mkdir(parents=True)
    tables_root = output / "tables"
    tables_root.mkdir()
    source_paths = {
        "development_market": inputs.development_typed_root / "crsp_security_day.parquet",
        "development_return": inputs.development_return_root / "crsp_security_day_return.parquet",
        "development_identity": inputs.development_typed_root / "crsp_identity_interval.parquet",
        "development_shares": inputs.development_typed_root / "crsp_shares_interval.parquet",
        "validation_market": inputs.validation_market_root / "crsp_security_day.parquet",
        "validation_return": inputs.validation_return_root / "crsp_security_day_return.parquet",
        "validation_identity": inputs.validation_market_root / "crsp_identity_interval.parquet",
        "validation_shares": inputs.validation_market_root / "crsp_shares_interval.parquet",
    }
    for label, path in source_paths.items():
        _require(path.is_file() and not path.is_symlink(), f"missing or unsafe {label}")
    _require_same_fields(
        source_paths["development_market"], source_paths["validation_market"], "market"
    )
    _require_same_fields(
        source_paths["development_return"], source_paths["validation_return"], "returns"
    )
    _require_same_fields(
        source_paths["development_identity"], source_paths["validation_identity"], "identity"
    )
    _require_same_fields(
        source_paths["development_shares"], source_paths["validation_shares"], "shares"
    )
    return_names = set(_field_schema(source_paths["validation_return"]).names)
    _require(
        {"raw_return_ex_distributions", "authoritative_total_return"}.issubset(return_names),
        "full predictor context would omit required historical return values",
    )
    _require(
        not any(
            token in name.lower()
            for name in return_names
            for token in ("forward_return", "future_return", "target", "label", "pnl")
        ),
        "a forbidden forward/target field entered the return context",
    )

    for name in (
        "ff_factor_daily_observation.parquet",
        "ff_factor_monthly_observation.parquet",
        "ps_liquidity_observation.parquet",
    ):
        development = inputs.development_auxiliary_root / name
        validation = inputs.validation_auxiliary_root / name
        _require(
            development.is_file()
            and validation.is_file()
            and not development.is_symlink()
            and not validation.is_symlink(),
            f"missing or unsafe factor input: {name}",
        )
        _require_same_fields(development, validation, name)

    spill = output / ".duckdb_spill"
    spill.mkdir()
    connection = duckdb.connect(":memory:")
    receipts: dict[str, Any] = {}
    try:
        connection.execute("SET preserve_insertion_order = false")
        connection.execute(f"SET threads = {FULL_CONTEXT_DUCKDB_THREADS}")
        connection.execute(
            f"SET memory_limit = '{FULL_CONTEXT_DUCKDB_MEMORY_LIMIT}'"
        )
        connection.execute("SET temp_directory = ?", [str(spill)])

        for label in ("market", "return"):
            development = source_paths[f"development_{label}"]
            validation = source_paths[f"validation_{label}"]
            dev_min, dev_max = connection.execute(
                f"SELECT min(calendar_date), max(calendar_date) FROM read_parquet('{_sql_path(development)}')"
            ).fetchone()
            ext_min, ext_max = connection.execute(
                f"SELECT min(calendar_date), max(calendar_date) FROM read_parquet('{_sql_path(validation)}')"
            ).fetchone()
            _require(dev_min >= DEVELOPMENT_CONTEXT_START, f"{label} begins before 2000")
            _require(dev_max < MARKET_EXTENSION_START, f"development {label} crosses extension")
            _require(ext_min >= MARKET_EXTENSION_START, f"validation {label} overlaps development")
            _require(ext_max <= LAST_PERMITTED_EXIT, f"validation {label} exceeds last exit")
            destination_name = (
                "crsp_security_day.parquet"
                if label == "market"
                else "crsp_security_day_return.parquet"
            )
            destination = tables_root / destination_name
            _emit(progress, f"full_predictor_{label}_started")
            _copy_query(
                connection,
                f"""
                SELECT * FROM (
                    SELECT * FROM read_parquet('{_sql_path(development)}')
                    WHERE calendar_date <= DATE '{LAST_SCORED_SIGNAL.isoformat()}'
                    UNION ALL
                    SELECT * FROM read_parquet('{_sql_path(validation)}')
                    WHERE calendar_date <= DATE '{LAST_SCORED_SIGNAL.isoformat()}'
                ) source
                ORDER BY permno, calendar_date
                """,
                destination,
            )
            _require(
                _logical_field_signature(destination)
                == _logical_field_signature(development),
                f"composed {label} field schema differs",
            )
            dev_overlap_differences = _table_difference_count(
                connection,
                f"SELECT * FROM read_parquet('{_sql_path(development)}')",
                f"SELECT * FROM read_parquet('{_sql_path(destination)}') WHERE calendar_date <= DATE '{dev_max.isoformat()}'",
            )
            _require(dev_overlap_differences == 0, f"development {label} overlap changed")
            rows, duplicates = _key_gate(
                connection, destination, ("permno", "calendar_date")
            )
            _require(duplicates == 0, f"composed {label} keys are duplicated")
            _sorted_unique_scan(destination, ("permno", "calendar_date"))
            receipts[destination_name] = {
                "rows": rows,
                "duplicate_keys": duplicates,
                "development_overlap_differences": dev_overlap_differences,
                "minimum_date": str(dev_min),
                "maximum_date": LAST_SCORED_SIGNAL.isoformat(),
                "sha256": _sha256(destination),
                "size_bytes": destination.stat().st_size,
            }
            _emit(progress, f"full_predictor_{label}_complete", rows=rows)

        for name, keys in (
            ("crsp_identity_interval.parquet", ("permno", "start_date", "end_date")),
            ("crsp_shares_interval.parquet", ("permno", "start_date", "end_date")),
        ):
            label = "identity" if name.startswith("crsp_identity") else "shares"
            development = source_paths[f"development_{label}"]
            validation = source_paths[f"validation_{label}"]
            destination = tables_root / name
            development_max_start = connection.execute(
                f"SELECT max(start_date) FROM read_parquet('{_sql_path(development)}')"
            ).fetchone()[0]
            _require(
                development_max_start is not None
                and development_max_start <= LAST_SCORED_SIGNAL,
                f"development {label} start-date scope is invalid",
            )
            dev_overlap_differences = _table_difference_count(
                connection,
                f"SELECT * FROM read_parquet('{_sql_path(development)}')",
                f"""
                SELECT * FROM read_parquet('{_sql_path(validation)}')
                WHERE start_date <= DATE '{development_max_start.isoformat()}'
                """,
            )
            _require(dev_overlap_differences == 0, f"development {label} overlap changed")
            _copy_query(
                connection,
                f"""
                SELECT * FROM read_parquet('{_sql_path(validation)}')
                WHERE start_date <= DATE '{LAST_SCORED_SIGNAL.isoformat()}'
                ORDER BY {','.join(keys)}
                """,
                destination,
            )
            rows, duplicates = _key_gate(connection, destination, keys)
            _require(duplicates == 0, f"composed {label} keys are duplicated")
            _sorted_unique_scan(destination, keys)
            receipts[name] = {
                "rows": rows,
                "duplicate_keys": duplicates,
                "development_overlap_differences": dev_overlap_differences,
                "maximum_start_date": LAST_SCORED_SIGNAL.isoformat(),
                "sha256": _sha256(destination),
                "size_bytes": destination.stat().st_size,
            }

        for name, date_column, keys in (
            ("ff_factor_daily_observation.parquet", "observation_date", ("observation_date",)),
            ("ff_factor_monthly_observation.parquet", "factor_month", ("factor_month",)),
            ("ps_liquidity_observation.parquet", "observation_month", ("observation_month",)),
        ):
            development = inputs.development_auxiliary_root / name
            validation = inputs.validation_auxiliary_root / name
            destination = tables_root / name
            dev_max = connection.execute(
                f"SELECT max({date_column}) FROM read_parquet('{_sql_path(development)}')"
            ).fetchone()[0]
            ext_min, ext_max = connection.execute(
                f"SELECT min({date_column}), max({date_column}) FROM read_parquet('{_sql_path(validation)}')"
            ).fetchone()
            _require(ext_min is not None and ext_max is not None, f"empty factor extension: {name}")
            _require(ext_min > dev_max, f"factor extension overlaps development: {name}")
            _require(
                ext_max <= LAST_SCORED_SIGNAL,
                f"factor extension exceeds predictor cutoff: {name}",
            )
            _copy_query(
                connection,
                f"""
                SELECT * FROM (
                    SELECT * FROM read_parquet('{_sql_path(development)}')
                    WHERE {date_column} <= DATE '{LAST_SCORED_SIGNAL.isoformat()}'
                    UNION ALL
                    SELECT * FROM read_parquet('{_sql_path(validation)}')
                    WHERE {date_column} <= DATE '{LAST_SCORED_SIGNAL.isoformat()}'
                ) source
                ORDER BY {','.join(keys)}
                """,
                destination,
            )
            dev_overlap_differences = _table_difference_count(
                connection,
                f"SELECT * FROM read_parquet('{_sql_path(development)}')",
                f"SELECT * FROM read_parquet('{_sql_path(destination)}') WHERE {date_column} <= DATE '{dev_max.isoformat()}'",
            )
            _require(dev_overlap_differences == 0, f"development factor overlap changed: {name}")
            rows, duplicates = _key_gate(connection, destination, keys)
            _require(duplicates == 0, f"composed factor keys are duplicated: {name}")
            _sorted_unique_scan(destination, keys)
            receipts[name] = {
                "rows": rows,
                "duplicate_keys": duplicates,
                "development_overlap_differences": dev_overlap_differences,
                "maximum_date": LAST_SCORED_SIGNAL.isoformat(),
                "sha256": _sha256(destination),
                "size_bytes": destination.stat().st_size,
            }

        key_differences = int(
            connection.execute(
                f"""
                SELECT count(*)
                FROM read_parquet('{_sql_path(tables_root / 'crsp_security_day.parquet')}') m
                FULL OUTER JOIN read_parquet('{_sql_path(tables_root / 'crsp_security_day_return.parquet')}') r
                USING (permno, calendar_date)
                WHERE m.permno IS NULL OR r.permno IS NULL
                """
            ).fetchone()[0]
        )
        _require(key_differences == 0, "full market and return context keys differ")
    finally:
        connection.close()
        try:
            spill.rmdir()
        except OSError:
            pass

    receipt = {
        "schema_version": 1,
        "status": "stage11_full_predictor_context_composed_no_forward_access",
        "scope": {
            "start": DEVELOPMENT_CONTEXT_START.isoformat(),
            "last_permitted_row_date": LAST_SCORED_SIGNAL.isoformat(),
            "internal_context_signal": INTERNAL_CONTEXT_SIGNAL.isoformat(),
            "scored_signals": REQUIRED_SCORED_SIGNALS,
        },
        "execution_resources": {
            "duckdb_threads": FULL_CONTEXT_DUCKDB_THREADS,
            "duckdb_memory_limit": FULL_CONTEXT_DUCKDB_MEMORY_LIMIT,
            "scientific_scope_or_formula_change": False,
        },
        "historical_returns": {
            "raw_return_ex_distributions_retained": True,
            "authoritative_total_return_retained": True,
            "use": "lagged_predictor_and_risk_input_only",
            "rows_after_last_signal": 0,
            "forward_label_target_columns": 0,
        },
        "development_overlap_differences": 0,
        "duplicate_keys": 0,
        "market_return_key_differences": 0,
        "tables": receipts,
        "authorizations": {
            "predictor_materialization_authorized": False,
            "labels_forward_returns_weights_trades_pnl_authorized": False,
            "final_2020_2025_access_authorized": False,
        },
    }
    _atomic_json(output / "full_predictor_context_receipt.json", receipt)
    return receipt


@contextmanager
def _frozen_auxiliary_scope(
    scope: ValidationUniverseAuxiliaryScope = DEFAULT_SCOPE,
) -> Iterator[dict[str, int]]:
    """Temporarily install exact lower/upper auxiliary date gates.

    SEC rows before the extension are rejected by the source gate before any
    payload column is decoded.  Compustat's legacy row parser has no lower
    bound hook, so its wrapper discards pre-extension normalized rows after
    applying the unchanged availability formula; those sources are orders of
    magnitude smaller than OptionMetrics/SEC and are reported separately.
    """

    _require(scope == DEFAULT_SCOPE, "only the frozen Stage 11 scope is supported")
    acquired = _AUXILIARY_SCOPE_LOCK.acquire(blocking=False)
    _require(acquired, "another validation-auxiliary scope is already active")
    compustat_prior = (compustat.LAST_SIGNAL, compustat.LAST_EXIT)
    sec_prior = sec.SCOPE_END
    sec_rows_prior = sec._date_first_rows
    fundamental_prior = compustat._fundamental_row
    counters = {
        "sec_pre_extension_rows_not_decoded": 0,
        "compustat_pre_extension_rows_decoded_then_discarded": 0,
    }

    class _MinimumAvailabilityGate:
        def __init__(self, delegate: Any) -> None:
            self.delegate = delegate
            self.pre_extension = 0

        def decide(
            self,
            columns: Mapping[str, list[str]],
            index: int,
        ) -> sec.GateDecision:
            decision = self.delegate.decide(columns, index)
            if (
                decision.include
                and decision.availability_date is not None
                and decision.availability_date < AUXILIARY_EXTENSION_START
            ):
                self.pre_extension += 1
                return sec.GateDecision(
                    False,
                    decision.observation_date,
                    decision.availability_date,
                    decision.availability_basis,
                    "post_scope",
                )
            return decision

    def scoped_sec_rows(
        path: Path,
        gate_columns: Sequence[str],
        payload_columns: Sequence[str],
        gate: Any,
        receipt: sec.SourceReceipt,
        *,
        max_source_records: int | None,
    ) -> Iterator[tuple[int, dict[str, str], sec.GateDecision]]:
        minimum_gate = _MinimumAvailabilityGate(gate)
        try:
            yield from sec_rows_prior(
                path,
                gate_columns,
                payload_columns,
                minimum_gate,
                receipt,
                max_source_records=max_source_records,
            )
        finally:
            receipt.post_scope_rows_not_decoded -= minimum_gate.pre_extension
            prior = int(getattr(receipt, "pre_extension_rows_not_decoded", 0))
            receipt.pre_extension_rows_not_decoded = prior + minimum_gate.pre_extension
            counters["sec_pre_extension_rows_not_decoded"] += minimum_gate.pre_extension

    def scoped_fundamental_row(**kwargs: Any) -> dict[str, Any] | None:
        row = fundamental_prior(**kwargs)
        if (
            row is not None
            and row["public_availability_date"] < AUXILIARY_EXTENSION_START
        ):
            counters["compustat_pre_extension_rows_decoded_then_discarded"] += 1
            return None
        return row

    compustat.LAST_SIGNAL = scope.last_scored_signal
    # Factor values after the last scored signal are validation outcomes.  The
    # live-unblind outcome layer, not this no-outcome normalizer, owns them.
    compustat.LAST_EXIT = scope.last_scored_signal
    sec.SCOPE_END = scope.last_scored_signal
    sec._date_first_rows = scoped_sec_rows
    compustat._fundamental_row = scoped_fundamental_row
    try:
        yield counters
    finally:
        compustat.LAST_SIGNAL, compustat.LAST_EXIT = compustat_prior
        sec.SCOPE_END = sec_prior
        sec._date_first_rows = sec_rows_prior
        compustat._fundamental_row = fundamental_prior
        _AUXILIARY_SCOPE_LOCK.release()


def _calendar_for_auxiliary(path: Path) -> AuxiliaryTradingCalendar:
    table = pq.read_table(path, columns=["calendar_date"], use_threads=False)
    return AuxiliaryTradingCalendar(table["calendar_date"].combine_chunks().to_pylist())


@contextmanager
def _optionmetrics_progress(progress: Progress | None) -> Iterator[None]:
    """Add periodic read progress without changing the frozen normalizer."""

    original = optionmetrics._open_text_batches

    class _Reader:
        def __init__(self, reader: Any, source: Path) -> None:
            self.reader = reader
            self.source = source.name
            self.rows = 0
            self.next_report = 5_000_000

        def __iter__(self) -> "_Reader":
            return self

        def __next__(self) -> pa.RecordBatch:
            batch = next(self.reader)
            self.rows += batch.num_rows
            if self.rows >= self.next_report:
                _emit(
                    progress,
                    "stage11_optionmetrics_source_progress",
                    source=self.source,
                    source_rows=self.rows,
                )
                self.next_report = ((self.rows // 5_000_000) + 1) * 5_000_000
            return batch

        def close(self) -> None:
            self.reader.close()

    def wrapped(path: Path, columns: Sequence[str], **kwargs: Any) -> _Reader:
        return _Reader(original(path, columns, **kwargs), path)

    optionmetrics._open_text_batches = wrapped
    try:
        yield
    finally:
        optionmetrics._open_text_batches = original


def _empty_segment_table(
    *,
    schema_document: Mapping[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    """Materialize the frozen zero-support segment table without reading raw rows."""

    table = table_from_rows(schema_document, "compustat_segment_observation", [])
    path = output_dir / "compustat_segment_observation.parquet"
    _require(not path.exists(), "segment target already exists")
    pq.write_table(table, path, compression="zstd", use_dictionary=True)
    return {
        "table": "compustat_segment_observation",
        "rows": 0,
        "path": path.name,
        "sha256": auxiliary_sha256(path),
        "size_bytes": path.stat().st_size,
        "support": "frozen_inactive_development_support_zero",
        "raw_segment_rows_read": 0,
    }


def _auxiliary_boundary_receipt(
    table_root: Path,
) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for path in sorted(table_root.glob("*.parquet")):
        file = pq.ParquetFile(path)
        schema = file.schema_arrow
        date_column = (
            "public_availability_date"
            if "public_availability_date" in schema.names
            else "observation_date"
            if "observation_date" in schema.names
            else "link_start_date"
            if path.name == "optionmetrics_crsp_link_interval.parquet"
            else "signal_date"
            if "signal_date" in schema.names
            else None
        )
        minimum: dt.date | None = None
        maximum: dt.date | None = None
        if date_column is not None and file.metadata.num_rows:
            for index in range(file.num_row_groups):
                values = file.read_row_group(index, columns=[date_column])[date_column]
                if values.null_count:
                    continue
                value = pc.max(values).as_py()
                minimum_value = pc.min(values).as_py()
                minimum = (
                    minimum_value if minimum is None else min(minimum, minimum_value)
                )
                maximum = value if maximum is None else max(maximum, value)
            _require(
                maximum is None or maximum <= LAST_SCORED_SIGNAL,
                f"auxiliary table exceeds its cutoff: {path.name}",
            )
            _require(
                minimum is None or minimum >= AUXILIARY_EXTENSION_START,
                f"auxiliary extension contains development rows: {path.name}",
            )
        checks[path.name] = {
            "rows": file.metadata.num_rows,
            "row_groups": file.num_row_groups,
            "maximum_gate_date": None if maximum is None else maximum.isoformat(),
            "minimum_gate_date": None if minimum is None else minimum.isoformat(),
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }
    segment = checks.get("compustat_segment_observation.parquet")
    _require(segment is not None and segment["rows"] == 0, "segment support was activated")
    return checks


def _checkpoint_outputs(table_root: Path, names: Sequence[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in names:
        path = table_root / name
        _require(path.is_file() and not path.is_symlink(), f"checkpoint output missing: {name}")
        file = pq.ParquetFile(path)
        result[name] = {
            "rows": file.metadata.num_rows,
            "row_groups": file.num_row_groups,
            "sha256": _sha256(path),
            "size_bytes": path.stat().st_size,
        }
    return result


def _factor_checkpoint_semantic_gate(table_root: Path) -> None:
    """Reject factor checkpoints containing observations after the signal cutoff."""

    for name, date_column in FACTOR_DATE_COLUMNS.items():
        path = table_root / name
        _require(
            path.is_file() and not path.is_symlink(),
            f"factor checkpoint output missing: {name}",
        )
        file = pq.ParquetFile(path)
        _require(date_column in file.schema_arrow.names, f"factor date column missing: {name}")
        maximum: dt.date | None = None
        for index in range(file.num_row_groups):
            values = file.read_row_group(index, columns=[date_column])[date_column]
            _require(values.null_count == 0, f"factor date contains nulls: {name}")
            if len(values):
                value = pc.max(values).as_py()
                maximum = value if maximum is None else max(maximum, value)
        _require(
            maximum is None or maximum <= LAST_SCORED_SIGNAL,
            f"factor checkpoint exceeds no-outcome cutoff: {name}",
        )


def _load_checkpoint(
    receipt_path: Path,
    table_root: Path,
    names: Sequence[str],
    semantic_validator: Callable[[Path], None] | None = None,
) -> dict[str, Any] | None:
    pending_path = receipt_path.with_name(receipt_path.stem + ".pending.json")
    if not receipt_path.is_file():
        if pending_path.is_file():
            pending = json.loads(pending_path.read_text("utf-8"))
            _require(isinstance(pending, dict), "invalid pending checkpoint receipt")
            expected = pending.get("outputs")
            existing = [name for name in names if (table_root / name).is_file()]
            if len(existing) == len(names):
                observed = _checkpoint_outputs(table_root, names)
                _require(observed == expected, "recovered checkpoint outputs changed")
                if semantic_validator is not None:
                    semantic_validator(table_root)
                os.replace(pending_path, receipt_path)
                return pending
            for name in existing:
                path = table_root / name
                observed = _checkpoint_outputs(table_root, (name,))[name]
                _require(
                    isinstance(expected, dict) and observed == expected.get(name),
                    "partial checkpoint output differs from pending receipt",
                )
                path.unlink()
            pending_path.unlink()
        _require(
            not any((table_root / name).exists() for name in names),
            f"unreceipted outputs exist for checkpoint {receipt_path.stem}",
        )
        return None
    value = json.loads(receipt_path.read_text("utf-8"))
    _require(
        isinstance(value, dict) and value.get("status") == "checkpoint_complete",
        f"invalid checkpoint receipt: {receipt_path.name}",
    )
    observed = _checkpoint_outputs(table_root, names)
    _require(observed == value.get("outputs"), f"checkpoint outputs changed: {receipt_path.stem}")
    if semantic_validator is not None:
        semantic_validator(table_root)
    return value


def _run_checkpoint(
    *,
    output_root: Path,
    table_root: Path,
    name: str,
    targets: Sequence[str],
    builder: Callable[[Path], Mapping[str, Any]],
    progress: Progress | None,
    semantic_validator: Callable[[Path], None] | None = None,
) -> dict[str, Any]:
    receipt_path = output_root / f"checkpoint_{name}.json"
    completed = _load_checkpoint(
        receipt_path,
        table_root,
        targets,
        semantic_validator,
    )
    if completed is not None:
        _emit(progress, f"{name}_checkpoint_reused", outputs=len(targets))
        return completed
    staging = output_root / f".staging_{name}"
    if staging.exists():
        _require(staging.is_dir() and not staging.is_symlink(), "unsafe checkpoint staging path")
        shutil.rmtree(staging)
    staging.mkdir()
    _emit(progress, f"{name}_checkpoint_started")
    try:
        details = dict(builder(staging))
        if semantic_validator is not None:
            semantic_validator(staging)
        staging_outputs = _checkpoint_outputs(staging, targets)
        receipt = {
            "schema_version": 1,
            "status": "checkpoint_complete",
            "checkpoint": name,
            "scope_start": AUXILIARY_EXTENSION_START.isoformat(),
            "scope_end": LAST_SCORED_SIGNAL.isoformat(),
            "outputs": staging_outputs,
            "details": details,
        }
        pending_path = receipt_path.with_name(receipt_path.stem + ".pending.json")
        _atomic_json(pending_path, receipt)
        for target in targets:
            source = staging / target
            _require(source.is_file(), f"checkpoint did not build {target}")
            destination = table_root / target
            _require(not destination.exists(), f"checkpoint target already exists: {target}")
            os.replace(source, destination)
        outputs = _checkpoint_outputs(table_root, targets)
        _require(outputs == staging_outputs, "checkpoint outputs changed during atomic move")
        os.replace(pending_path, receipt_path)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    _emit(progress, f"{name}_checkpoint_complete", outputs=len(targets))
    return receipt


def _trim_table_after_development(
    *,
    source_path: Path,
    development_path: Path,
    date_column: str,
    maximum_date: dt.date,
) -> dict[str, Any]:
    """Rewrite a factor table to a bounded post-development extension."""

    development = pq.read_table(development_path, columns=[date_column], use_threads=False)
    _require(development.num_rows > 0, f"empty development factor table: {development_path.name}")
    last_development = pc.max(development[date_column]).as_py()
    table = pq.read_table(source_path, use_threads=False)
    date_type = table.schema.field(date_column).type
    retained = table.filter(
        pc.and_(
            pc.greater(table[date_column], pa.scalar(last_development, type=date_type)),
            pc.less_equal(table[date_column], pa.scalar(maximum_date, type=date_type)),
        )
    )
    temporary = source_path.with_name(source_path.name + ".trimmed")
    pq.write_table(
        retained,
        temporary,
        compression="zstd",
        use_dictionary=True,
        write_statistics=True,
    )
    os.replace(temporary, source_path)
    return {
        "last_development_date": last_development.isoformat(),
        "maximum_extension_date": maximum_date.isoformat(),
        "extension_rows": retained.num_rows,
    }


def build_validation_auxiliary_normalization(
    *,
    raw_root: Path,
    development_auxiliary_root: Path,
    trading_calendar_path: Path,
    validation_identity_path: Path,
    scored_universe_path: Path,
    internal_context_universe_path: Path,
    output_root: Path,
    progress: Progress | None = None,
    scope: ValidationUniverseAuxiliaryScope = DEFAULT_SCOPE,
) -> dict[str, Any]:
    """Build only the post-development auxiliary extension, resumably.

    Verified development normalizations are reused byte-for-byte.  The large
    OptionMetrics tables begin on 2014-11-01, and SEC's lower bound is applied
    before payload decoding.  Every independently expensive family has an
    authenticated checkpoint, so rerunning the command reuses completed work.
    Segment raw data is never opened.
    """

    _require(scope == DEFAULT_SCOPE, "only the frozen Stage 11 scope is supported")
    load_policy()
    development_binding = _authenticate_development_auxiliary_root(
        development_auxiliary_root
    )
    for path, label in (
        (raw_root, "raw root"),
        (development_auxiliary_root, "development auxiliary root"),
        (trading_calendar_path, "trading calendar"),
        (validation_identity_path, "validation identity"),
        (scored_universe_path, "scored universe"),
        (internal_context_universe_path, "internal context universe"),
    ):
        _require(path.exists() and not path.is_symlink(), f"missing or unsafe {label}")
    output = output_root.resolve()
    if not output.exists():
        output.mkdir(parents=True)
    _require(output.is_dir() and not output.is_symlink(), "unsafe auxiliary output")
    table_root = output / "tables"
    table_root.mkdir(exist_ok=True)
    schema_document = load_schema_document(AUXILIARY_SCHEMA_PATH)
    final_receipt_path = output / RECEIPT_FILE
    if final_receipt_path.is_file():
        final = json.loads(final_receipt_path.read_text("utf-8"))
        _require(isinstance(final, dict), "invalid auxiliary completion receipt")
        observed = _auxiliary_boundary_receipt(table_root)
        _require(observed == final.get("boundary_checks"), "completed auxiliary outputs changed")
        _emit(progress, "stage11_auxiliary_complete_receipt_reused", tables=len(observed))
        return final

    sec_targets = tuple(f"{name}.parquet" for name in sec.TABLE_SPECS)
    checkpoints: dict[str, Any] = {}
    with _frozen_auxiliary_scope(scope) as scope_counters:
        checkpoints["sec_restatement"] = _run_checkpoint(
            output_root=output,
            table_root=table_root,
            name="sec_restatement",
            targets=sec_targets,
            progress=progress,
            builder=lambda staging: build_sec_restatement_family(
                raw_root=raw_root,
                trading_calendar_path=trading_calendar_path,
                crsp_identity_path=validation_identity_path,
                staging_dir=staging,
                batch_rows=250_000,
                progress=progress,
            ),
        )

        checkpoints["ccm_links"] = _run_checkpoint(
            output_root=output,
            table_root=table_root,
            name="ccm_links",
            targets=("ccm_link_interval.parquet",),
            progress=progress,
            builder=lambda staging: compustat.build_ccm_link_intervals(
                raw_path=raw_root / "ccm_link_table.csv",
                output_dir=staging,
                schema_document=schema_document,
            ),
        )
        for table_name, filename in (
            ("compustat_annual_observation", "comp_funda_ccm.csv"),
            ("compustat_quarterly_observation", "comp_fundq_ccm.csv"),
        ):
            checkpoints[table_name] = _run_checkpoint(
                output_root=output,
                table_root=table_root,
                name=table_name,
                targets=(f"{table_name}.parquet",),
                progress=progress,
                builder=lambda staging, name=table_name, source=filename: compustat.build_fundamental_table(
                    name=name,
                    raw_path=raw_root / source,
                    output_dir=staging,
                    schema_document=schema_document,
                    calendar=_calendar_for_auxiliary(trading_calendar_path),
                    sec_filing_path=table_root / "sec_filing_observation.parquet",
                ),
            )
        checkpoints["segments_frozen_inactive"] = _run_checkpoint(
            output_root=output,
            table_root=table_root,
            name="segments_frozen_inactive",
            targets=("compustat_segment_observation.parquet",),
            progress=progress,
            builder=lambda staging: _empty_segment_table(
                schema_document=schema_document,
                output_dir=staging,
            ),
        )

        factor_targets = tuple(FACTOR_DATE_COLUMNS)

        def build_factor_extension(staging: Path) -> Mapping[str, Any]:
            outputs = compustat.build_factor_tables(
                raw_root=raw_root,
                output_dir=staging,
                schema_document=schema_document,
                calendar=_calendar_for_auxiliary(trading_calendar_path),
            )
            trims = {
                name: _trim_table_after_development(
                    source_path=staging / name,
                    development_path=development_auxiliary_root / name,
                    date_column=date_column,
                    maximum_date=LAST_SCORED_SIGNAL,
                )
                for name, date_column in FACTOR_DATE_COLUMNS.items()
            }
            return {"untrimmed_outputs": outputs, "trims": trims}

        checkpoints["factor_extension"] = _run_checkpoint(
            output_root=output,
            table_root=table_root,
            name="factor_extension",
            targets=factor_targets,
            progress=progress,
            builder=build_factor_extension,
            semantic_validator=_factor_checkpoint_semantic_gate,
        )

        option_settings = lambda staging: OptionMetricsSettings(
            raw_root=raw_root,
            crsp_identity_path=validation_identity_path,
            staging_dir=staging,
            scope_start=AUXILIARY_EXTENSION_START,
            scope_end=scope.last_scored_signal,
        )

        def build_option_names_links(staging: Path) -> Mapping[str, Any]:
            settings = option_settings(staging)
            names = build_name_intervals(settings)
            links = build_crsp_links(settings, names)
            return {"name_rows": names.num_rows, "link_rows": links.num_rows}

        with _optionmetrics_progress(progress):
            checkpoints["optionmetrics_names_links"] = _run_checkpoint(
                output_root=output,
                table_root=table_root,
                name="optionmetrics_names_links",
                targets=(
                    "optionmetrics_name_interval.parquet",
                    "optionmetrics_crsp_link_interval.parquet",
                ),
                progress=progress,
                builder=build_option_names_links,
            )
            checkpoints["optionmetrics_volume"] = _run_checkpoint(
                output_root=output,
                table_root=table_root,
                name="optionmetrics_volume",
                targets=("optionmetrics_option_volume_observation.parquet",),
                progress=progress,
                builder=lambda staging: stream_option_volume(option_settings(staging)),
            )
            checkpoints["optionmetrics_historical_volatility"] = _run_checkpoint(
                output_root=output,
                table_root=table_root,
                name="optionmetrics_historical_volatility",
                targets=("optionmetrics_historical_volatility_observation.parquet",),
                progress=progress,
                builder=lambda staging: stream_historical_volatility(option_settings(staging)),
            )

    checkpoints["scored_auxiliary_linkage"] = _run_checkpoint(
        output_root=output,
        table_root=table_root,
        name="scored_auxiliary_linkage",
        targets=(SCORED_LINKAGE_FILE,),
        progress=progress,
        builder=lambda staging: build_auxiliary_signal_linkage(
            signal_universe_path=scored_universe_path,
            ccm_link_path=table_root / "ccm_link_interval.parquet",
            optionmetrics_link_path=table_root / "optionmetrics_crsp_link_interval.parquet",
            output_path=staging / SCORED_LINKAGE_FILE,
        ),
    )
    checkpoints["context_auxiliary_linkage"] = _run_checkpoint(
        output_root=output,
        table_root=table_root,
        name="context_auxiliary_linkage",
        targets=(CONTEXT_LINKAGE_FILE,),
        progress=progress,
        builder=lambda staging: build_auxiliary_signal_linkage(
            signal_universe_path=internal_context_universe_path,
            ccm_link_path=table_root / "ccm_link_interval.parquet",
            optionmetrics_link_path=table_root / "optionmetrics_crsp_link_interval.parquet",
            output_path=staging / CONTEXT_LINKAGE_FILE,
        ),
    )
    scored_linkage = checkpoints["scored_auxiliary_linkage"]["details"]
    context_linkage = checkpoints["context_auxiliary_linkage"]["details"]
    boundary_checks = _auxiliary_boundary_receipt(table_root)
    _require(
        scored_linkage["source_spine_rows"] == scored_linkage["rows"]
        and context_linkage["source_spine_rows"] == context_linkage["rows"],
        "auxiliary linkage did not preserve a signal spine",
    )
    receipt = {
        "schema_version": 1,
        "status": "stage11_validation_auxiliary_normalization_built_no_outcomes",
        "contract_id": CONTRACT_ID,
        "scope": {
            "extension_start": AUXILIARY_EXTENSION_START.isoformat(),
            "internal_context_signal": INTERNAL_CONTEXT_SIGNAL.isoformat(),
            "first_scored_signal": FIRST_SCORED_SIGNAL.isoformat(),
            "last_scored_signal": LAST_SCORED_SIGNAL.isoformat(),
            "factor_last_observation": LAST_SCORED_SIGNAL.isoformat(),
            "first_final_entry": FIRST_FINAL_ENTRY.isoformat(),
        },
        "checkpoint_receipts": checkpoints,
        "lower_bound_gate_counts": scope_counters,
        "linkage": {
            "scored": scored_linkage,
            "internal_context": context_linkage,
            "context_signal_rows_published": 0,
        },
        "boundary_checks": boundary_checks,
        "segment_support": {
            "status": "frozen_inactive_development_support_zero",
            "normalized_rows": 0,
            "raw_rows_read": 0,
            "validation_support_activation_permitted": False,
        },
        "bindings": {
            "policy_sha256": _sha256(POLICY_PATH),
            "development_auxiliary_root": str(development_auxiliary_root.resolve()),
            "development_auxiliary": development_binding,
            "universe_formula_sha256": _sha256(Path(frozen_universe.__file__)),
            "compustat_normalizer_sha256": _sha256(Path(compustat.__file__)),
            "sec_normalizer_sha256": _sha256(Path(sec.__file__)),
        },
        "authorizations": {
            "features_or_model_fitting_authorized": False,
            "labels_forward_returns_weights_trades_pnl_authorized": False,
            "final_2020_2025_access_authorized": False,
        },
    }
    _atomic_json(final_receipt_path, receipt)
    return receipt
