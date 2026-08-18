"""Primary Stage 7 development-label materialization.

The module owns content-addressed build identity, exact spine alignment, and
one-signal-month label construction.  Cache/publication mechanics are kept
separate so an interrupted full build can resume at a complete signal month.
It never fits a model, constructs a portfolio, calculates P&L, or reads beyond
the frozen development cutoff.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from portfolio_bot.data.secure_publication import (
    SecureStagingDirectory,
)
from portfolio_bot.features.label_math import (
    FACTOR_NAMES,
    MISSING_BETA,
    NOT_LONG_ELIGIBLE,
    PRESENT,
    FactorObservation,
    ForwardLeg,
    ForwardOutcome,
    assign_split_membership,
    build_month_targets,
    derive_label_statuses,
    evaluate_forward_outcome,
    month_label_available_date,
)
from portfolio_bot.features.label_return_router import (
    LabelReturnRouter,
    compute_label_intervals_sha256,
)
from portfolio_bot.features.label_store_contract import (
    DEFAULT_POLICY,
    PROJECT_ROOT,
    published_schemas,
    sha256_file,
    validate_label_store_contract,
)
from portfolio_bot.features.label_store_inputs import (
    LabelStoreInputs,
    reauthenticate_label_inputs,
)
from portfolio_bot.features.phase_03_contract import canonical_json_bytes


_FEATURE_TABLE = "feature_store_raw.parquet"
_UNIVERSE_TABLE = "crsp_signal_date_universe.parquet"
_BETA_NAMES = tuple(f"beta_label_{name}" for name in FACTOR_NAMES)
_CODE_PATHS = (
    "src/portfolio_bot/data/secure_publication.py",
    "src/portfolio_bot/features/phase_03_contract.py",
    "src/portfolio_bot/features/label_store_contract.py",
    "src/portfolio_bot/features/label_store_inputs.py",
    "src/portfolio_bot/features/label_math.py",
    "src/portfolio_bot/features/label_return_router.py",
    "src/portfolio_bot/features/label_store_materialization.py",
    "src/portfolio_bot/features/label_store_quality.py",
    "src/portfolio_bot/features/label_store_publication.py",
    "scripts/build_stage_07_label_store.py",
)


class LabelStoreMaterializationError(RuntimeError):
    """Raised when Stage 7 cannot be reproduced from authenticated inputs."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise LabelStoreMaterializationError(message)


def _toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        value = tomllib.load(handle)
    _require(isinstance(value, dict), f"Expected Stage 7 TOML object: {path}")
    return value


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text("utf-8"))
    _require(isinstance(value, dict), f"Expected Stage 7 JSON object: {path}")
    return value


def _project_path(value: str | Path, project_root: Path = PROJECT_ROOT) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (project_root / path).resolve()


def _schema_sha256(schema: pa.Schema) -> str:
    return hashlib.sha256(schema.serialize().to_pybytes()).hexdigest()


@dataclass(frozen=True)
class LabelInputReceipt:
    gate: str
    name: str
    sha256: str
    schema_sha256: str
    size_bytes: int
    rows: int
    row_groups: int

    def as_dict(self) -> dict[str, object]:
        return {
            "gate": self.gate,
            "name": self.name,
            "sha256": self.sha256,
            "schema_sha256": self.schema_sha256,
            "size_bytes": self.size_bytes,
            "rows": self.rows,
            "row_groups": self.row_groups,
        }


@dataclass(frozen=True)
class LabelStoreBuildIdentity:
    contract_id: str
    policy_id: str
    build_id: str
    content_fingerprint: str
    configuration_hashes: tuple[tuple[str, str], ...]
    code_hashes: tuple[tuple[str, str], ...]
    input_receipts: tuple[LabelInputReceipt, ...]

    @property
    def configuration_hash_map(self) -> dict[str, str]:
        return dict(self.configuration_hashes)

    @property
    def code_hash_map(self) -> dict[str, str]:
        return dict(self.code_hashes)

    @property
    def input_receipts_sha256(self) -> str:
        payload = [receipt.as_dict() for receipt in self.input_receipts]
        return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _configuration_paths(
    policy: Mapping[str, Any],
    *,
    policy_path: Path,
    project_root: Path,
) -> tuple[Path, ...]:
    values = (
        policy_path,
        _project_path(policy["physical_rules"]["path"], project_root),
        _project_path(policy["semantic_contract"]["policy_path"], project_root),
        _project_path(policy["semantic_contract"]["verified_pointer"], project_root),
        _project_path(policy["verified_feature_store"]["policy_path"], project_root),
        _project_path(policy["verified_feature_store"]["verified_pointer"], project_root),
        _project_path(policy["verified_feature_store"]["primary_manifest"], project_root),
        _project_path(policy["database"]["paths_config"], project_root),
        _project_path(policy["database"]["relocation_receipt"], project_root),
        *(
            _project_path(value, project_root)
            for name, value in policy["inputs"]["verified_pointers"].items()
            if not name.endswith("_sha256")
        ),
    )
    unique = tuple(sorted(set(values), key=lambda path: path.as_posix()))
    for path in unique:
        _require(
            path.is_file() and not path.is_symlink(),
            f"Stage 7 identity configuration is absent or symlinked: {path}",
        )
    return unique


def _relative_identity_name(path: Path, project_root: Path) -> str:
    try:
        return path.relative_to(project_root).as_posix()
    except ValueError:
        return path.as_posix()


def compute_label_store_build_identity(
    inputs: LabelStoreInputs,
    *,
    policy_path: Path = DEFAULT_POLICY,
    project_root: Path = PROJECT_ROOT,
) -> LabelStoreBuildIdentity:
    """Bind the primary build to exact configuration, code, and six inputs."""

    policy_path = policy_path.resolve()
    project_root = project_root.resolve()
    contract = validate_label_store_contract(policy_path)
    policy = _toml(policy_path)
    _require(inputs.hashes_authenticated, "Stage 7 build identity needs full hashes")
    _require(
        tuple(inputs.tables) == tuple(contract["input_names"]),
        "Stage 7 identity input set differs",
    )
    configuration_hashes = tuple(
        (
            _relative_identity_name(path, project_root),
            sha256_file(path),
        )
        for path in _configuration_paths(
            policy,
            policy_path=policy_path,
            project_root=project_root,
        )
    )
    code_hashes: list[tuple[str, str]] = []
    for relative in _CODE_PATHS:
        path = project_root / relative
        _require(
            path.is_file() and not path.is_symlink(),
            f"Stage 7 production dependency is missing: {relative}",
        )
        code_hashes.append((relative, sha256_file(path)))
    receipts = tuple(
        LabelInputReceipt(
            gate=table.gate,
            name=name,
            sha256=table.sha256,
            schema_sha256=table.schema_sha256,
            size_bytes=table.size_bytes,
            rows=table.rows,
            row_groups=table.row_groups,
        )
        for name, table in inputs.tables.items()
    )
    fingerprint_payload = {
        "schema_version": 1,
        "contract_id": contract["contract_id"],
        "policy_id": contract["policy_id"],
        "configuration_hashes": dict(configuration_hashes),
        "code_hashes": dict(code_hashes),
        "input_receipts": [receipt.as_dict() for receipt in receipts],
    }
    fingerprint = hashlib.sha256(
        canonical_json_bytes(fingerprint_payload)
    ).hexdigest()
    return LabelStoreBuildIdentity(
        contract_id=str(contract["contract_id"]),
        policy_id=str(contract["policy_id"]),
        build_id=f"phase03-labels-{fingerprint[:24]}",
        content_fingerprint=fingerprint,
        configuration_hashes=configuration_hashes,
        code_hashes=tuple(code_hashes),
        input_receipts=receipts,
    )


def validate_label_store_build_identity(
    inputs: LabelStoreInputs,
    identity: LabelStoreBuildIdentity,
    *,
    policy_path: Path = DEFAULT_POLICY,
    project_root: Path = PROJECT_ROOT,
) -> None:
    observed = compute_label_store_build_identity(
        inputs,
        policy_path=policy_path,
        project_root=project_root,
    )
    _require(observed == identity, "Stage 7 build identity differs")


def reauthenticate_label_store_build_identity(
    inputs: LabelStoreInputs,
    identity: LabelStoreBuildIdentity,
    *,
    policy_path: Path = DEFAULT_POLICY,
    project_root: Path = PROJECT_ROOT,
) -> None:
    reauthenticate_label_inputs(inputs)
    validate_label_store_build_identity(
        inputs,
        identity,
        policy_path=policy_path,
        project_root=project_root,
    )


@dataclass(frozen=True)
class DevelopmentCalendarRow:
    signal_date: dt.date
    holding_month: str
    entry_date: dt.date
    exit_date: dt.date


@dataclass(frozen=True)
class DevelopmentLabelContext:
    feature_spine: pa.Table
    calendar_rows: tuple[DevelopmentCalendarRow, ...]
    sessions: tuple[dt.date, ...]
    factors: Mapping[dt.date, FactorObservation]
    month_slices: Mapping[dt.date, tuple[int, int]]
    long_intervals: pa.Table


def _holding_month(value: object) -> str:
    if isinstance(value, str):
        parsed = dt.datetime.strptime(value, "%Y-%m")
        normalized = parsed.strftime("%Y-%m")
        _require(value == normalized, f"Invalid holding month: {value}")
        return normalized
    if isinstance(value, dt.datetime):
        value = value.date()
    if isinstance(value, dt.date):
        return value.strftime("%Y-%m")
    raise LabelStoreMaterializationError(f"Invalid holding month: {value!r}")


def _array_equal(left: pa.ChunkedArray, right: pa.ChunkedArray) -> bool:
    if len(left) != len(right):
        return False
    equal = pc.equal(left, right)
    return bool(pc.all(pc.fill_null(equal, False)).as_py())


def _month_slices(signal_dates: pa.ChunkedArray) -> dict[dt.date, tuple[int, int]]:
    values = signal_dates.combine_chunks().to_pylist()
    result: dict[dt.date, tuple[int, int]] = {}
    start = 0
    while start < len(values):
        signal = values[start]
        end = start + 1
        while end < len(values) and values[end] == signal:
            end += 1
        _require(signal not in result, "Stage 7 spine signal date is noncontiguous")
        result[signal] = (start, end)
        start = end
    return result


def _validate_spine_order(table: pa.Table) -> None:
    signals = table["signal_date"].combine_chunks().to_numpy(zero_copy_only=False)
    permnos = table["permno"].combine_chunks().to_numpy(zero_copy_only=False)
    if len(signals) < 2:
        return
    signal_days = signals.astype("datetime64[D]").astype(np.int64)
    violation = (signal_days[1:] < signal_days[:-1]) | (
        (signal_days[1:] == signal_days[:-1])
        & (permnos[1:] <= permnos[:-1])
    )
    _require(not bool(np.any(violation)), "Stage 7 spine key is not strict sorted")


def load_development_label_context(
    inputs: LabelStoreInputs,
    *,
    policy_path: Path = DEFAULT_POLICY,
) -> DevelopmentLabelContext:
    """Load only the bounded, authenticated context needed by Stage 7."""

    contract = validate_label_store_contract(policy_path)
    feature_spec = inputs.tables[_FEATURE_TABLE]
    universe_spec = inputs.tables[_UNIVERSE_TABLE]
    feature = pq.read_table(
        inputs.path(_FEATURE_TABLE),
        columns=list(feature_spec.projected_columns),
        use_threads=False,
    ).combine_chunks()
    universe = pq.read_table(
        inputs.path(_UNIVERSE_TABLE),
        columns=list(universe_spec.projected_columns),
        use_threads=False,
    ).combine_chunks()
    _require(
        feature.num_rows == universe.num_rows == contract["required_audit_rows"],
        "Stage 7 feature/universe spine row count differs",
    )
    _validate_spine_order(feature)
    for name in (
        "signal_date",
        "permno",
        "permco",
        "long_eligible",
        "short_eligible",
    ):
        _require(
            _array_equal(feature[name], universe[name]),
            f"Stage 7 feature/universe field differs: {name}",
        )
    _require(
        universe["issuer_line_selected"].null_count == 0
        and _array_equal(
            universe["issuer_line_selected"],
            universe["long_eligible"],
        ),
        "Stage 7 issuer-line selection differs from long eligibility",
    )
    _require(
        universe["future_outcome_filter_applied"].null_count == 0
        and not bool(pc.any(universe["future_outcome_filter_applied"]).as_py()),
        "Stage 7 universe contains a future-outcome filter",
    )

    raw_calendar = inputs.read_development_calendar().combine_chunks()
    _require(
        all(column.null_count == 0 for column in raw_calendar.columns),
        "Stage 7 bounded development calendar contains nulls",
    )
    calendar_rows = tuple(
        DevelopmentCalendarRow(
            signal_date=row["signal_date"],
            holding_month=_holding_month(row["holding_month"]),
            entry_date=row["entry_date"],
            exit_date=row["exit_date"],
        )
        for row in raw_calendar.to_pylist()
    )
    _require(
        len(calendar_rows) == contract["required_signal_dates"]
        and tuple(row.signal_date for row in calendar_rows)
        == tuple(sorted({row.signal_date for row in calendar_rows})),
        "Stage 7 development calendar differs",
    )
    slices = _month_slices(feature["signal_date"])
    _require(
        tuple(slices) == tuple(row.signal_date for row in calendar_rows),
        "Stage 7 feature/calendar signal domain differs",
    )

    trading = inputs.read_development_trading_calendar()
    _require(
        trading["calendar_date"].null_count == 0,
        "Stage 7 bounded trading calendar contains nulls",
    )
    sessions = tuple(trading["calendar_date"].to_pylist())
    _require(
        sessions == tuple(sorted(set(sessions))),
        "Stage 7 trading calendar is not unique sorted",
    )
    factor_rows = inputs.read_development_factors().to_pylist()
    factors: dict[dt.date, FactorObservation] = {}
    for row in factor_rows:
        date = row["observation_date"]
        _require(date not in factors, "Duplicate Stage 7 factor date")
        factors[date] = FactorObservation(
            calendar_date=date,
            rf=float(row["rf"]),
            mktrf=float(row["mktrf"]),
            smb=float(row["smb"]),
            hml=float(row["hml"]),
            rmw=float(row["rmw"]),
            cma=float(row["cma"]),
            umd=float(row["umd"]),
            public_availability_date=row["public_availability_date"],
        )
    _require(
        tuple(factors) == sessions,
        "Stage 7 factor and trading calendars differ",
    )

    calendar_by_signal = {row.signal_date: row for row in calendar_rows}
    interval_signals: list[dt.date] = []
    interval_permnos: list[int] = []
    interval_entries: list[dt.date] = []
    interval_exits: list[dt.date] = []
    long_total = 0
    for signal, (start, end) in slices.items():
        month = feature.slice(start, end - start)
        long_mask = month["long_eligible"].combine_chunks()
        selected = month.filter(long_mask)
        row = calendar_by_signal[signal]
        count = selected.num_rows
        long_total += count
        interval_signals.extend([signal] * count)
        interval_permnos.extend(int(value) for value in selected["permno"].to_pylist())
        interval_entries.extend([row.entry_date] * count)
        interval_exits.extend([row.exit_date] * count)
    _require(
        long_total == contract["required_long_eligible_rows"],
        "Stage 7 long interval count differs",
    )
    long_intervals = pa.table(
        {
            "signal_date": pa.array(interval_signals, type=pa.date32()),
            "permno": pa.array(interval_permnos, type=pa.int64()),
            "entry_date": pa.array(interval_entries, type=pa.date32()),
            "exit_date": pa.array(interval_exits, type=pa.date32()),
        }
    )
    return DevelopmentLabelContext(
        feature_spine=feature,
        calendar_rows=calendar_rows,
        sessions=sessions,
        factors=factors,
        month_slices=slices,
        long_intervals=long_intervals,
    )


@dataclass(frozen=True)
class LabelMonthTables:
    signal_date: dt.date
    audit: pa.Table
    lineage: pa.Table
    summary: pa.Table


def _required_sessions(
    sessions: Sequence[dt.date],
    entry_date: dt.date,
    exit_date: dt.date,
) -> tuple[dt.date, ...]:
    result = tuple(date for date in sessions if entry_date < date <= exit_date)
    _require(
        result and result[-1] == exit_date,
        "Stage 7 holding interval has no verified exit session",
    )
    return result


def _finite_betas(row: Mapping[str, Any]) -> tuple[float | None, ...]:
    return tuple(
        None if row[name] is None else float(row[name])
        for name in _BETA_NAMES
    )


def _return_legs(
    routed_returns: pa.Table,
    *,
    signal_date: dt.date,
) -> dict[int, dict[dt.date, list[ForwardLeg]]]:
    result: dict[int, dict[dt.date, list[ForwardLeg]]] = {}
    for row in routed_returns.to_pylist():
        _require(
            row["signal_date"] == signal_date,
            "Stage 7 routed return signal differs",
        )
        permno = int(row["permno"])
        date = row["calendar_date"]
        by_date = result.setdefault(permno, {})
        by_date.setdefault(date, []).append(
            ForwardLeg(
                calendar_date=date,
                span_start_date=row["return_span_start_date"],
                span_end_date=row["return_span_end_date"],
                return_value=(
                    None
                    if row["authoritative_total_return"] is None
                    else float(row["authoritative_total_return"])
                ),
                outcome_availability_date=row["outcome_availability_date"],
                outcome_class=str(row["return_leg_class"]),
                return_history_usable=bool(row["return_history_usable"]),
                # Row-level return_history_authorized is a constant-false
                # pointer guard. The authenticated verified pointer, not that
                # marker, supplies authorization to this builder.
                verified_return_pointer_authorized=True,
                covered_session_count=row["covered_trading_sessions"],
                delisting_status=str(row["delisting_treatment_status"]),
            )
        )
    return result


def _canonical_mean(values: Sequence[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None]
    if not finite:
        return None
    result = math.fsum(finite) / len(finite)
    return 0.0 if result == 0.0 else float(result)


def _empty_outcome(status: str, expected_count: int) -> ForwardOutcome:
    return ForwardOutcome(
        status=status,
        residual_sum=None,
        excess_sum=None,
        available_date=None,
        expected_session_count=expected_count,
        used_session_count=0,
        terminal_date=None,
        contributions=(),
    )


def materialize_label_signal_month(
    *,
    feature_month: pa.Table,
    routed_returns: pa.Table,
    calendar: DevelopmentCalendarRow,
    sessions: Sequence[dt.date],
    factors: Mapping[dt.date, FactorObservation],
    rules: Mapping[str, Any],
) -> LabelMonthTables:
    """Construct one complete, deterministically ranked signal month."""

    schemas = published_schemas(rules)
    rows = feature_month.to_pylist()
    _require(rows, "Stage 7 signal month is empty")
    _require(
        all(row["signal_date"] == calendar.signal_date for row in rows),
        "Stage 7 feature month signal differs",
    )
    permnos = [int(row["permno"]) for row in rows]
    _require(
        permnos == sorted(set(permnos)),
        "Stage 7 feature month is not unique sorted by PERMNO",
    )
    required = _required_sessions(sessions, calendar.entry_date, calendar.exit_date)
    legs_by_permno = _return_legs(
        routed_returns,
        signal_date=calendar.signal_date,
    )
    long_permnos = {
        int(row["permno"]) for row in rows if bool(row["long_eligible"])
    }
    _require(
        set(legs_by_permno).issubset(long_permnos),
        "Stage 7 return router emitted a nonlong security",
    )

    outcomes: list[ForwardOutcome] = []
    forward_valid: list[bool] = []
    beta_complete: list[bool] = []
    residuals: list[float | None] = []
    excesses: list[float | None] = []
    primary_valid: list[bool] = []
    long_flags = [bool(row["long_eligible"]) for row in rows]

    for row, long_eligible in zip(rows, long_flags, strict=True):
        betas = _finite_betas(row)
        complete = len(betas) == 6 and all(
            value is not None and math.isfinite(value) for value in betas
        )
        beta_complete.append(complete)
        if not long_eligible:
            outcome = _empty_outcome(NOT_LONG_ELIGIBLE, len(required))
            valid_forward = False
        else:
            permno = int(row["permno"])
            offered = legs_by_permno.get(permno, {})
            outcome = evaluate_forward_outcome(
                calendar.entry_date,
                calendar.exit_date,
                required,
                betas,
                offered,
                factors,
            )
            if complete:
                valid_forward = outcome.status == PRESENT
            else:
                forward_only = evaluate_forward_outcome(
                    calendar.entry_date,
                    calendar.exit_date,
                    required,
                    (0.0,) * 6,
                    offered,
                    factors,
                )
                valid_forward = forward_only.status == PRESENT
                _require(
                    outcome.status == MISSING_BETA,
                    "Stage 7 missing beta precedence differs",
                )
        outcomes.append(outcome)
        forward_valid.append(valid_forward)
        is_primary = long_eligible and outcome.status == PRESENT
        primary_valid.append(is_primary)
        residuals.append(outcome.residual_sum if is_primary else None)
        excesses.append(outcome.excess_sum if is_primary else None)

    targets = build_month_targets(
        residuals,
        excesses,
        primary_valid,
        long_flags,
        minimum_count=int(rules["targets"]["minimum_valid_labels_per_signal"]),
    )
    security_dates = [outcome.available_date for outcome in outcomes]
    month_available = month_label_available_date(
        security_dates,
        primary_valid,
        long_flags,
    )
    zero_mad = targets.magnitude_status == "MISSING_ZERO_MAD"
    audit_rows: list[dict[str, Any]] = []
    lineage_rows: list[dict[str, Any]] = []
    for index, (row, outcome) in enumerate(zip(rows, outcomes, strict=True)):
        primary_status, magnitude_status = derive_label_statuses(
            outcome.status,
            targets.month_status,
            zero_mad=zero_mad,
        )
        contributions = outcome.contributions if outcome.status == PRESENT else ()
        max_outcome_date = (
            max(item.outcome_availability_date for item in contributions)
            if contributions
            else None
        )
        betas = _finite_betas(row)
        audit_rows.append(
            {
                "signal_date": calendar.signal_date,
                "permno": int(row["permno"]),
                "permco": int(row["permco"]),
                "long_eligible": bool(row["long_eligible"]),
                "short_eligible": bool(row["short_eligible"]),
                "holding_month": calendar.holding_month,
                "entry_date": calendar.entry_date,
                "exit_date": calendar.exit_date,
                **{
                    name: betas[position]
                    for position, name in enumerate(_BETA_NAMES)
                },
                "beta_complete": beta_complete[index],
                "expected_leg_count": len(required),
                "used_leg_count": outcome.used_session_count,
                "first_used_leg_date": (
                    contributions[0].calendar_date if contributions else None
                ),
                "last_used_leg_date": (
                    contributions[-1].calendar_date if contributions else None
                ),
                "terminal_delisting_used": outcome.terminal_date is not None,
                "terminal_delisting_date": outcome.terminal_date,
                "max_outcome_availability_date": max_outcome_date,
                "security_label_available_date": outcome.available_date,
                "month_label_available_date": month_available,
                "unranked_excess_return": excesses[index],
                "unranked_residual_return": residuals[index],
                "residual_rank_numerator": targets.residual_rank_numerators[index],
                "excess_rank_numerator": targets.excess_rank_numerators[index],
                "residual_return_rank": targets.residual_rank[index],
                "excess_return_rank": targets.excess_rank[index],
                "residual_return_mad_standardized": (
                    targets.residual_mad_standardized[index]
                ),
                "valid_primary_count_in_month": targets.valid_count,
                "outcome_status": outcome.status,
                "primary_label_status": primary_status,
                "magnitude_label_status": magnitude_status,
            }
        )
        for ordinal, item in enumerate(contributions, start=1):
            lineage_rows.append(
                {
                    "signal_date": calendar.signal_date,
                    "permno": int(row["permno"]),
                    "calendar_date": item.calendar_date,
                    "leg_ordinal": ordinal,
                    "return_leg_class": item.return_leg_class,
                    "authoritative_total_return": item.return_value,
                    "rf": item.rf,
                    **{
                        name: item.factor_values[position]
                        for position, name in enumerate(FACTOR_NAMES)
                    },
                    "systematic_component": item.systematic_component,
                    "excess_return_contribution": item.excess_contribution,
                    "residual_return_contribution": item.residual_contribution,
                    "outcome_availability_date": item.outcome_availability_date,
                    "factor_public_availability_date": (
                        item.factor_public_availability_date
                    ),
                    "terminal_delisting": item.terminal_delisting,
                }
            )

    audit = pa.Table.from_pylist(
        audit_rows,
        schema=schemas["development_label_audit.parquet"],
    )
    lineage = pa.Table.from_pylist(
        lineage_rows,
        schema=schemas["development_label_leg_lineage.parquet"],
    )
    eligible_count = sum(long_flags)
    beta_count = sum(
        long and complete
        for long, complete in zip(long_flags, beta_complete, strict=True)
    )
    valid_forward_count = sum(
        long and valid
        for long, valid in zip(long_flags, forward_valid, strict=True)
    )
    valid_count = sum(primary_valid)
    terminal_count = sum(
        valid and outcome.terminal_date is not None
        for valid, outcome in zip(primary_valid, outcomes, strict=True)
    )
    delayed_count = sum(
        valid
        and outcome.available_date is not None
        and outcome.available_date > calendar.exit_date
        for valid, outcome in zip(primary_valid, outcomes, strict=True)
    )
    summary = pa.Table.from_pylist(
        [
            {
                "signal_date": calendar.signal_date,
                "holding_month": calendar.holding_month,
                "entry_date": calendar.entry_date,
                "exit_date": calendar.exit_date,
                "eligible_security_count": eligible_count,
                "beta_complete_count": beta_count,
                "valid_forward_count": valid_forward_count,
                "valid_primary_count": valid_count,
                "minimum_required_count": int(
                    rules["targets"]["minimum_valid_labels_per_signal"]
                ),
                "primary_month_valid": targets.month_status == PRESENT,
                "terminal_delisting_label_count": terminal_count,
                "delayed_availability_count": delayed_count,
                "month_label_available_date": month_available,
                "residual_median": targets.residual_median,
                "residual_mad": targets.residual_mad,
                "residual_rank_numerator_sum": (
                    targets.residual_rank_numerator_sum
                ),
                "excess_rank_numerator_sum": (
                    targets.excess_rank_numerator_sum
                ),
                "residual_rank_mean": _canonical_mean(targets.residual_rank),
                "excess_rank_mean": _canonical_mean(targets.excess_rank),
                "month_status": targets.month_status,
            }
        ],
        schema=schemas["development_label_month_summary.parquet"],
    )
    return LabelMonthTables(
        signal_date=calendar.signal_date,
        audit=audit,
        lineage=lineage,
        summary=summary,
    )


def project_development_labels(
    audit: pa.Table,
    *,
    rules: Mapping[str, Any],
) -> pa.Table:
    """Project the exact long-only handoff without dropping invalid labels."""

    schema = published_schemas(rules)["development_labels.parquet"]
    filtered = audit.filter(audit["long_eligible"])
    return filtered.select(schema.names).cast(schema)


@dataclass(frozen=True)
class LabelCacheFileReceipt:
    path: str
    sha256: str
    size_bytes: int
    rows: int
    row_groups: int
    schema_sha256: str

    def as_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "rows": self.rows,
            "row_groups": self.row_groups,
            "schema_sha256": self.schema_sha256,
        }


@dataclass(frozen=True)
class LabelMonthReceipt:
    path: Path
    build_id: str
    signal_date: dt.date
    ordinal: int
    audit_rows: int
    long_rows: int
    lineage_rows: int
    summary: Mapping[str, Any]
    files: tuple[LabelCacheFileReceipt, ...]
    success_sha256: str
    reused_existing: bool


@dataclass(frozen=True)
class LabelStoreCacheResult:
    identity: LabelStoreBuildIdentity
    month_receipts: tuple[LabelMonthReceipt, ...]
    audit_rows: int
    long_rows: int
    lineage_rows: int
    final_identity_reauthenticated: bool


def _parquet_bytes(
    table: pa.Table,
    *,
    row_group_size: int,
    compression: str,
    dictionary: bool,
) -> tuple[bytes, LabelCacheFileReceipt]:
    sink = pa.BufferOutputStream()
    pq.write_table(
        table,
        sink,
        row_group_size=row_group_size,
        compression=compression,
        use_dictionary=dictionary,
        version="2.6",
        write_statistics=True,
    )
    payload = sink.getvalue().to_pybytes()
    parquet = pq.ParquetFile(pa.BufferReader(payload))
    receipt = LabelCacheFileReceipt(
        path="",
        sha256=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        rows=parquet.metadata.num_rows,
        row_groups=parquet.metadata.num_row_groups,
        schema_sha256=_schema_sha256(parquet.schema_arrow),
    )
    return payload, receipt


def _default_label_rules() -> dict[str, Any]:
    policy = _toml(DEFAULT_POLICY)
    return _json(_project_path(policy["physical_rules"]["path"]))


def _validate_cache_identity(
    inputs: LabelStoreInputs,
    identity: LabelStoreBuildIdentity,
) -> None:
    _require(
        inputs.hashes_authenticated,
        "Stage 7 cache requires authenticated inputs",
    )
    _require(
        len(identity.content_fingerprint) == 64
        and all(character in "0123456789abcdef" for character in identity.content_fingerprint)
        and identity.build_id
        == f"phase03-labels-{identity.content_fingerprint[:24]}",
        "Stage 7 cache build identity is malformed",
    )
    observed = tuple(
        LabelInputReceipt(
            gate=table.gate,
            name=name,
            sha256=table.sha256,
            schema_sha256=table.schema_sha256,
            size_bytes=table.size_bytes,
            rows=table.rows,
            row_groups=table.row_groups,
        )
        for name, table in inputs.tables.items()
    )
    _require(
        observed == identity.input_receipts,
        "Stage 7 cache input receipts differ from build identity",
    )


def _strict_keys(
    table: pa.Table,
    names: Sequence[str],
) -> bool:
    keys = list(
        zip(
            *(table[name].to_pylist() for name in names),
            strict=True,
        )
    )
    return all(left < right for left, right in zip(keys, keys[1:]))


def _validate_month_tables_for_cache(
    tables: LabelMonthTables,
    *,
    rules: Mapping[str, Any],
) -> None:
    schemas = published_schemas(rules)
    expected = {
        "audit": schemas["development_label_audit.parquet"],
        "lineage": schemas["development_label_leg_lineage.parquet"],
        "summary": schemas["development_label_month_summary.parquet"],
    }
    for name, schema in expected.items():
        table = getattr(tables, name)
        _require(
            isinstance(table, pa.Table)
            and table.schema.equals(schema, check_metadata=True),
            f"Stage 7 cache {name} schema differs",
        )
        for field in schema:
            if not field.nullable:
                _require(
                    table[field.name].null_count == 0,
                    f"Stage 7 cache {name} has a null non-null field",
                )
            if pa.types.is_floating(field.type):
                _require(
                    all(
                        value is None or math.isfinite(float(value))
                        for value in table[field.name].to_pylist()
                    ),
                    f"Stage 7 cache {name} has a nonfinite value",
                )
    _require(tables.audit.num_rows > 0, "Stage 7 cache audit is empty")
    _require(
        tables.summary.num_rows == 1,
        "Stage 7 cache must contain exactly one month summary",
    )
    audit_signals = set(tables.audit["signal_date"].to_pylist())
    lineage_signals = set(tables.lineage["signal_date"].to_pylist())
    summary_row = tables.summary.to_pylist()[0]
    _require(
        audit_signals == {tables.signal_date}
        and summary_row["signal_date"] == tables.signal_date
        and (not lineage_signals or lineage_signals == {tables.signal_date}),
        "Stage 7 cache signal-date reconciliation differs",
    )
    _require(
        _strict_keys(tables.audit, ("signal_date", "permno"))
        and _strict_keys(
            tables.lineage,
            ("signal_date", "permno", "calendar_date"),
        ),
        "Stage 7 cache table key is not strict sorted",
    )
    audit_rows = tables.audit.to_pylist()
    long_rows = sum(bool(row["long_eligible"]) for row in audit_rows)
    valid_rows = sum(
        bool(row["long_eligible"]) and row["outcome_status"] == PRESENT
        for row in audit_rows
    )
    used_legs = sum(int(row["used_leg_count"]) for row in audit_rows)
    _require(
        int(summary_row["eligible_security_count"]) == long_rows
        and int(summary_row["valid_primary_count"]) == valid_rows
        and used_legs == tables.lineage.num_rows,
        "Stage 7 cache summary/count reconciliation differs",
    )
    for name in (
        "holding_month",
        "entry_date",
        "exit_date",
        "month_label_available_date",
    ):
        _require(
            set(tables.audit[name].to_pylist()) == {summary_row[name]},
            f"Stage 7 cache audit/summary field differs: {name}",
        )


def _safe_cache_destination(
    destination: Path,
    *,
    database_root: Path,
) -> Path:
    root = database_root.resolve()
    candidate = Path(destination)
    _require(
        candidate.is_absolute(),
        "Stage 7 cache destination must be absolute",
    )
    candidate = Path(candidate.absolute())
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise LabelStoreMaterializationError(
            "Stage 7 cache destination escaped database root"
        ) from exc
    _require(
        relative.parts
        and relative.parts[0] == "cache"
        and ".." not in relative.parts,
        "Stage 7 cache destination is outside cache lifecycle",
    )
    return candidate


def materialize_label_month_cache(
    *,
    inputs: LabelStoreInputs,
    identity: LabelStoreBuildIdentity,
    ordinal: int,
    tables: LabelMonthTables,
    destination: Path,
    row_group_size: int = 131_072,
    compression: str = "zstd",
    dictionary: bool = True,
    rules: Mapping[str, Any] | None = None,
) -> LabelMonthReceipt:
    """Publish one immutable, restartable signal-month cache bundle."""

    _require(
        isinstance(ordinal, int) and not isinstance(ordinal, bool) and ordinal >= 0,
        "Stage 7 month ordinal is invalid",
    )
    _require(row_group_size > 0, "Stage 7 cache row-group size is invalid")
    _validate_cache_identity(inputs, identity)
    effective_rules = _default_label_rules() if rules is None else rules
    _validate_month_tables_for_cache(tables, rules=effective_rules)
    destination = _safe_cache_destination(
        destination,
        database_root=inputs.database_root,
    )
    names = {
        "label_audit.parquet": tables.audit,
        "label_leg_lineage.parquet": tables.lineage,
    }
    payloads: dict[str, bytes] = {}
    receipts: dict[str, LabelCacheFileReceipt] = {}
    for name, table in names.items():
        payload, receipt = _parquet_bytes(
            table,
            row_group_size=row_group_size,
            compression=compression,
            dictionary=dictionary,
        )
        payloads[name] = payload
        receipts[name] = LabelCacheFileReceipt(
            path=name,
            sha256=receipt.sha256,
            size_bytes=receipt.size_bytes,
            rows=receipt.rows,
            row_groups=receipt.row_groups,
            schema_sha256=receipt.schema_sha256,
        )
    summary = tables.summary.to_pylist()[0]
    json_summary = {
        key: value.isoformat()
        if isinstance(value, (dt.date, dt.datetime))
        else value
        for key, value in summary.items()
    }
    success = {
        "schema_version": 1,
        "contract_id": identity.contract_id,
        "policy_id": identity.policy_id,
        "build_id": identity.build_id,
        "status": "LABEL_MONTH_CACHE_COMPLETE",
        "ordinal": ordinal,
        "signal_date": tables.signal_date.isoformat(),
        "summary": json_summary,
        "table_receipts": {
            name: receipts[name].as_dict() for name in sorted(receipts)
        },
    }
    success_payload = canonical_json_bytes(success)
    success_hash = hashlib.sha256(success_payload).hexdigest()
    expected_files = (
        "_SUCCESS.json",
        "label_audit.parquet",
        "label_leg_lineage.parquet",
    )
    with SecureStagingDirectory.create(
        trusted_root=inputs.database_root,
        destination=destination,
        expected_files=expected_files,
        label=f"Stage 7 label month {ordinal:03d}",
    ) as stage:
        for name in ("label_audit.parquet", "label_leg_lineage.parquet"):
            with stage.open_exclusive_binary(name) as stream:
                stream.write(payloads[name])
        with stage.open_exclusive_binary("_SUCCESS.json") as stream:
            stream.write(success_payload)
        expected_hashes = {
            "_SUCCESS.json": success_hash,
            **{name: receipt.sha256 for name, receipt in receipts.items()},
        }
        snapshot = stage.snapshot(expected_hashes=expected_hashes)
        bundle = stage.publish(expected_hashes=snapshot.hashes)
    return LabelMonthReceipt(
        path=destination,
        build_id=identity.build_id,
        signal_date=tables.signal_date,
        ordinal=ordinal,
        audit_rows=tables.audit.num_rows,
        long_rows=int(pc.sum(tables.audit["long_eligible"]).as_py() or 0),
        lineage_rows=tables.lineage.num_rows,
        summary=summary,
        files=tuple(receipts[name] for name in sorted(receipts)),
        success_sha256=success_hash,
        reused_existing=bundle.reused_existing,
    )


def materialize_label_store_cache(
    *,
    inputs: LabelStoreInputs,
    identity: LabelStoreBuildIdentity,
    context: DevelopmentLabelContext,
    return_router: LabelReturnRouter,
    rules: Mapping[str, Any],
    cache_root: Path,
    progress: Any | None = None,
    policy_path: Path = DEFAULT_POLICY,
) -> LabelStoreCacheResult:
    """Materialize all 178 development signal months as immutable cache units."""

    policy_path = policy_path.resolve()
    validate_label_store_build_identity(
        inputs,
        identity,
        policy_path=policy_path,
    )
    policy = _toml(policy_path)
    bound_rules = _json(_project_path(policy["physical_rules"]["path"]))
    _require(
        canonical_json_bytes(rules) == canonical_json_bytes(bound_rules),
        "Stage 7 in-memory rules differ from the identity-bound artifact",
    )
    canonical_context = load_development_label_context(
        inputs,
        policy_path=policy_path,
    )
    _require(
        context.feature_spine.equals(
            canonical_context.feature_spine,
            check_metadata=True,
        )
        and context.calendar_rows == canonical_context.calendar_rows
        and context.sessions == canonical_context.sessions
        and dict(context.factors) == dict(canonical_context.factors)
        and dict(context.month_slices)
        == dict(canonical_context.month_slices)
        and context.long_intervals.equals(
            canonical_context.long_intervals,
            check_metadata=True,
        ),
        "Stage 7 in-memory context differs from authenticated inputs",
    )
    context = canonical_context
    rules = bound_rules
    _require(
        isinstance(return_router, LabelReturnRouter)
        and return_router.build_id == identity.build_id
        and return_router.source_fingerprint == identity.content_fingerprint
        and return_router.intervals_sha256
        == compute_label_intervals_sha256(context.long_intervals),
        "Stage 7 return-router build identity differs",
    )
    return_router.gate()
    root = _safe_cache_destination(
        Path(cache_root) / "month_materialization" / identity.build_id,
        database_root=inputs.database_root,
    )
    receipts: list[LabelMonthReceipt] = []
    for ordinal, calendar in enumerate(context.calendar_rows):
        start, end = context.month_slices[calendar.signal_date]
        feature_month = context.feature_spine.slice(start, end - start)
        routed_batches = tuple(
            return_router.iter_signal_batches(calendar.signal_date)
        )
        routed = (
            pa.Table.from_batches(routed_batches)
            if routed_batches
            else pa.Table.from_batches(
                [],
                schema=return_router.schema,
            )
        )
        tables = materialize_label_signal_month(
            feature_month=feature_month,
            routed_returns=routed,
            calendar=calendar,
            sessions=context.sessions,
            factors=context.factors,
            rules=rules,
        )
        destination = root / (
            f"month-{ordinal:06d}-{calendar.signal_date.isoformat()}"
        )
        receipt = materialize_label_month_cache(
            inputs=inputs,
            identity=identity,
            ordinal=ordinal,
            tables=tables,
            destination=destination,
            rules=rules,
        )
        receipts.append(receipt)
        if progress is not None:
            progress(
                {
                    "months_complete": ordinal + 1,
                    "months_total": len(context.calendar_rows),
                    "audit_rows": sum(item.audit_rows for item in receipts),
                    "long_rows": sum(item.long_rows for item in receipts),
                    "lineage_rows": sum(item.lineage_rows for item in receipts),
                }
            )
    audit_rows = sum(item.audit_rows for item in receipts)
    long_rows = sum(item.long_rows for item in receipts)
    lineage_rows = sum(item.lineage_rows for item in receipts)
    _require(
        len(receipts) == 178
        and audit_rows == 1_263_048
        and long_rows == 286_628,
        "Stage 7 cache completion counts differ",
    )
    return_router.gate()
    reauthenticate_label_store_build_identity(
        inputs,
        identity,
        policy_path=policy_path,
    )
    return LabelStoreCacheResult(
        identity=identity,
        month_receipts=tuple(receipts),
        audit_rows=audit_rows,
        long_rows=long_rows,
        lineage_rows=lineage_rows,
        final_identity_reauthenticated=True,
    )
