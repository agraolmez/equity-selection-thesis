"""Publish the development-only signal-date universe as an immutable bundle."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from portfolio_bot.data.phase_02_crsp_typed_backbone_core import (
    TableSpec,
    load_schema_registry,
)
from portfolio_bot.data.phase_02_signal_date_universe import (
    POLICY_PATH,
    PROJECT_ROOT,
    SCHEMA_PATH,
    SignalUniverseError,
    compute_universe,
    load_settings,
)


PRIMARY_POINTER_PATH = (
    PROJECT_ROOT
    / "artifacts/phase_02_canonical_panel/LATEST_SIGNAL_DATE_UNIVERSE_PRIMARY.json"
)
CONTRACT_ID = "phase02-crsp-b03263a970e52fbd"
STATUS = "signal_date_universe_primary_passed_pending_independent_verification"


@dataclass(frozen=True)
class PublicationSettings:
    output_database_root: Path
    build_id: str
    interim_dir: Path
    quality_dir: Path
    artifact_dir: Path
    is_production: bool


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_bytes(_canonical_json(value))
    os.replace(temporary, path)


def publication_settings(output_database_root: Path | None = None) -> PublicationSettings:
    core = load_settings()
    output_root = (output_database_root or core.database_root).resolve()
    production = output_root == core.database_root.resolve()
    identity = {
        "contract_id": CONTRACT_ID,
        "policy_sha256": _sha256(POLICY_PATH),
        "schema_sha256": _sha256(SCHEMA_PATH),
        "core_sha256": _sha256(PROJECT_ROOT / "src/portfolio_bot/data/phase_02_signal_date_universe.py"),
        "publication_sha256": _sha256(Path(__file__)),
        "typed_build_id": core.typed_root.name,
        "return_build_id": core.return_root.name,
        "calendar_sha256": _sha256(core.calendar_path),
    }
    semantic_hash = hashlib.sha256(_canonical_json(identity)).hexdigest()
    build_id = f"signal-universe-{semantic_hash[:24]}"
    relative = Path("phase_02_canonical_panel") / CONTRACT_ID / "signal_date_universe" / build_id
    artifact_dir = (
        PROJECT_ROOT / "artifacts/phase_02_canonical_panel/signal_date_universes" / build_id
        if production
        else output_root / "artifacts/phase_02_canonical_panel/signal_date_universes" / build_id
    )
    return PublicationSettings(
        output_database_root=output_root,
        build_id=build_id,
        interim_dir=output_root / "interim" / relative,
        quality_dir=output_root / "quality" / relative,
        artifact_dir=artifact_dir,
        is_production=production,
    )


def _validate_table(path: Path, spec: TableSpec, expected_rows: int) -> dict[str, Any]:
    parquet = pq.ParquetFile(path)
    if not parquet.schema_arrow.equals(spec.arrow_schema(), check_metadata=True):
        raise SignalUniverseError("Published universe Arrow schema differs from contract")
    if parquet.metadata.num_rows != expected_rows:
        raise SignalUniverseError("Published universe row count differs from computation")
    previous: tuple[Any, int] | None = None
    scanned = 0
    nullable_violations = 0
    for batch in parquet.iter_batches(
        columns=["signal_date", "permno"], batch_size=262_144
    ):
        dates = batch.column(0).to_numpy(zero_copy_only=False).astype("datetime64[D]").astype(np.int64)
        permno = batch.column(1).to_numpy(zero_copy_only=False).astype(np.int64)
        if len(dates):
            if previous is not None and (int(dates[0]), int(permno[0])) <= previous:
                raise SignalUniverseError("Universe primary key is not strictly increasing")
            descending = (dates[1:] < dates[:-1]) | (
                (dates[1:] == dates[:-1]) & (permno[1:] <= permno[:-1])
            )
            if np.any(descending):
                raise SignalUniverseError("Universe primary key is duplicated or unsorted")
            previous = (int(dates[-1]), int(permno[-1]))
        scanned += len(dates)
    if scanned != expected_rows:
        raise SignalUniverseError("Universe full key scan did not conserve rows")
    table = pq.read_table(
        path,
        columns=[
            "availability_not_after_signal",
            "return_observations_after_signal_count",
            "future_outcome_filter_applied",
            "long_eligible",
            "short_eligible",
            "issuer_line_selected",
        ],
    )
    if pc.any(pc.invert(table["availability_not_after_signal"])).as_py():
        nullable_violations += 1
    if pc.any(pc.not_equal(table["return_observations_after_signal_count"], 0)).as_py():
        nullable_violations += 1
    if pc.any(table["future_outcome_filter_applied"]).as_py():
        nullable_violations += 1
    if pc.any(pc.and_(table["short_eligible"], pc.invert(table["long_eligible"]))).as_py():
        nullable_violations += 1
    if pc.any(pc.and_(table["long_eligible"], pc.invert(table["issuer_line_selected"]))).as_py():
        nullable_violations += 1
    if nullable_violations:
        raise SignalUniverseError("Published universe failed point-in-time or implication gates")
    return {
        "path": path.name,
        "rows": expected_rows,
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _quality_tables(columns: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    signal = columns["signal_date"]
    exchange = columns["primary_exchange"]
    share = columns["share_type"]
    base = np.asarray(columns["base_eligible"], dtype=bool)
    long = np.asarray(columns["long_eligible"], dtype=bool)
    short = np.asarray(columns["short_eligible"], dtype=bool)
    by_signal: dict[Any, list[int]] = defaultdict(lambda: [0, 0, 0, 0])
    by_group: dict[tuple[int, str, str], list[int]] = defaultdict(lambda: [0, 0, 0, 0])
    for index, day in enumerate(signal):
        values = (1, int(base[index]), int(long[index]), int(short[index]))
        target = by_signal[day]
        group = by_group[(day.year, str(exchange[index]), str(share[index]))]
        for position, value in enumerate(values):
            target[position] += value
            group[position] += value
    signal_rows = [
        {
            "signal_date": day.isoformat(),
            "rows": values[0],
            "base_eligible": values[1],
            "long_eligible": values[2],
            "short_eligible": values[3],
        }
        for day, values in sorted(by_signal.items())
    ]
    group_rows = [
        {
            "signal_year": key[0],
            "primary_exchange": key[1],
            "share_type": key[2],
            "rows": values[0],
            "base_eligible": values[1],
            "long_eligible": values[2],
            "short_eligible": values[3],
        }
        for key, values in sorted(by_group.items())
    ]
    reason_rows: list[dict[str, Any]] = []
    for family in ("base", "long", "short"):
        counts: Counter[str] = Counter()
        for rendered in columns[f"{family}_reason_codes"]:
            counts.update(str(rendered).split("|"))
        reason_rows.extend(
            {"family": family, "reason_code": reason, "row_count": count}
            for reason, count in sorted(counts.items())
        )
    return signal_rows, group_rows, reason_rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise SignalUniverseError(f"Refusing empty quality receipt {path.name}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build(
    *,
    output_database_root: Path | None = None,
    publish_pointer: bool = True,
) -> dict[str, Any]:
    settings = publication_settings(output_database_root)
    if publish_pointer and not settings.is_production:
        raise SignalUniverseError("A rehearsal universe build cannot move production authority")
    finals = (settings.interim_dir, settings.quality_dir, settings.artifact_dir)
    if any(path.exists() for path in finals):
        raise SignalUniverseError(f"Universe build {settings.build_id} already exists")
    stages = tuple(path.with_name(".staging-" + settings.build_id) for path in finals)
    if any(path.exists() for path in stages):
        raise SignalUniverseError("A prior universe staging directory requires inspection")
    for path in stages:
        path.mkdir(parents=True)
    interim_stage, quality_stage, artifact_stage = stages
    try:
        print("universe gate 1/4: computing point-in-time screens and issuer selection", flush=True)
        result = compute_universe()
        registry = load_schema_registry(SCHEMA_PATH)
        spec = registry.tables["crsp_signal_date_universe"]
        table = pa.Table.from_pydict(result.columns, schema=spec.arrow_schema())
        universe_path = interim_stage / "crsp_signal_date_universe.parquet"
        pq.write_table(
            table,
            universe_path,
            compression="zstd",
            use_dictionary=True,
            write_statistics=True,
            row_group_size=131_072,
        )
        del table
        print("universe gate 2/4: validating full output and aggregate quality", flush=True)
        output = _validate_table(universe_path, spec, int(result.summary["rows"]))
        signal_rows, group_rows, reason_rows = _quality_tables(result.columns)
        _write_csv(quality_stage / "counts_by_signal.csv", signal_rows)
        _write_csv(quality_stage / "counts_by_year_exchange_share_type.csv", group_rows)
        _write_csv(quality_stage / "reason_counts.csv", reason_rows)
        checks = [
            ("row_conservation", "pass", result.summary["rows"]),
            ("unique_signal_date_permno", "pass", 0),
            ("identity_join_cardinality_one", "pass", result.summary["identity_matched_rows"]),
            ("availability_after_signal", "pass", 0),
            ("future_outcome_filter_applied", "pass", 0),
            ("short_not_long_violations", "pass", 0),
            ("selected_issuer_duplicates", "pass", 0),
        ]
        with (quality_stage / "checks.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["check", "status", "detail"])
            writer.writerows(checks)

        summary = {
            "schema_version": 1,
            "status": STATUS,
            "build_id": settings.build_id,
            "contract_id": CONTRACT_ID,
            "scope": {"first_signal": "2000-01-31", "last_signal": "2014-10-31"},
            "summary": result.summary,
            "output": output,
            "quality_receipts": {
                name: {
                    "sha256": _sha256(quality_stage / name),
                    "size_bytes": (quality_stage / name).stat().st_size,
                }
                for name in (
                    "checks.csv",
                    "counts_by_signal.csv",
                    "counts_by_year_exchange_share_type.csv",
                    "reason_counts.csv",
                )
            },
            "bindings": {
                "policy_sha256": _sha256(POLICY_PATH),
                "schema_sha256": _sha256(SCHEMA_PATH),
                "core_sha256": _sha256(PROJECT_ROOT / "src/portfolio_bot/data/phase_02_signal_date_universe.py"),
                "publication_sha256": _sha256(Path(__file__)),
                "typed_verified_pointer_sha256": _sha256(PROJECT_ROOT / "artifacts/phase_02_canonical_panel/LATEST_CRSP_TYPED_BACKBONE_VERIFIED.json"),
                "return_verified_pointer_sha256": _sha256(PROJECT_ROOT / "artifacts/phase_02_canonical_panel/LATEST_CRSP_RETURN_RECONCILIATION_VERIFIED.json"),
                "calendar_verified_pointer_sha256": _sha256(PROJECT_ROOT / "artifacts/phase_02_canonical_panel/LATEST_CALENDAR_VERIFIED.json"),
            },
            "authorizations": {
                "signal_date_universe_authorized": False,
                "independent_verification_required": True,
                "features_labels_models_portfolios_pnl_performance_or_backtests_authorized": False,
                "validation_or_final_test_access_authorized": False,
            },
        }
        (interim_stage / "_SUCCESS.json").write_bytes(_canonical_json(summary))
        (quality_stage / "summary.json").write_bytes(_canonical_json(summary))
        summary["quality_receipts"]["summary.json"] = {
            "sha256": _sha256(quality_stage / "summary.json"),
            "size_bytes": (quality_stage / "summary.json").stat().st_size,
        }
        manifest = {
            **summary,
            "interim_path": str(settings.interim_dir.relative_to(settings.output_database_root)),
            "quality_path": str(settings.quality_dir.relative_to(settings.output_database_root)),
        }
        (artifact_stage / "manifest.json").write_bytes(_canonical_json(manifest))
        hashes = [
            f"{output['sha256']}  crsp_signal_date_universe.parquet",
            *(
                f"{receipt['sha256']}  {name}"
                for name, receipt in sorted(summary["quality_receipts"].items())
            ),
        ]
        (artifact_stage / "output_hashes.sha256").write_text(
            "\n".join(hashes) + "\n", encoding="utf-8"
        )
        artifact_record = (
            str(settings.artifact_dir.relative_to(PROJECT_ROOT))
            if settings.is_production
            else str(settings.artifact_dir.relative_to(settings.output_database_root))
        )
        pointer = {
            "schema_version": 1,
            "status": STATUS,
            "build_id": settings.build_id,
            "contract_id": CONTRACT_ID,
            "artifact_path": artifact_record,
            "interim_path": manifest["interim_path"],
            "quality_path": manifest["quality_path"],
            "manifest_sha256": _sha256(artifact_stage / "manifest.json"),
            "signal_date_universe_authorized": False,
        }
        print("universe gate 3/4: publishing immutable primary bundle", flush=True)
        moved: list[tuple[Path, Path]] = []
        try:
            for stage, final in zip(stages, finals, strict=True):
                os.replace(stage, final)
                moved.append((stage, final))
            if publish_pointer:
                _atomic_json(PRIMARY_POINTER_PATH, pointer)
        except Exception:
            for stage, final in reversed(moved):
                if final.exists() and not stage.exists():
                    os.replace(final, stage)
            raise
        print("universe gate 4/4: primary passed; independent verification required", flush=True)
        return pointer
    except Exception:
        raise
