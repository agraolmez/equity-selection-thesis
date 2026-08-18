"""Independently verify the Phase 2 CRSP supporting-source profile.

The verifier is intentionally standalone: it never imports the primary
supporting compiler or publisher.  It authenticates their published bytes,
rereads all five raw sources through a separate scanner, reconstructs every
aggregate quality table, and produces a content-bound verification report.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as dt
import hashlib
import io
import json
import os
import re
import stat
import tempfile
import tomllib
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROFILE_CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "phase_02_canonicalization"
    / "crsp_supporting_sources_development_v1.toml"
)
DEFAULT_PATHS_CONFIG = PROJECT_ROOT / "configs" / "paths.toml"
DEFAULT_PHASE_CONFIG = PROJECT_ROOT / "configs" / "phase_02_canonical_panel.toml"
DEFAULT_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "phase_02_canonical_panel"
DEFAULT_PRIMARY_POINTER = (
    DEFAULT_ARTIFACT_ROOT / "LATEST_CRSP_SUPPORTING_PROFILE_PRIMARY.json"
)
DEFAULT_SCRATCH_ROOT = (
    PROJECT_ROOT / "runs" / "phase_02_canonical_panel" / "supporting_verifier_scratch"
)
PRIMARY_STATUS = "supporting_profile_primary_passed_pending_independent_verification"
PASS_STATUS = "verified_passed"
FAIL_STATUS = "verification_failed"
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")

SOURCE_ORDER = (
    "crsp_names_ciz.csv",
    "crsp_shares_outstanding_ciz.csv",
    "crsp_distribution_ciz.csv",
    "crsp_delisting_ciz.csv",
    "crsp_stock_header_ciz.csv",
)

CSV_SCHEMAS: dict[str, tuple[str, ...]] = {
    "categorical_domains.csv": (
        "source",
        "column",
        "raw_value",
        "row_count",
        "is_blank",
        "classification",
    ),
    "checks.csv": (
        "check_id",
        "source",
        "category",
        "metric",
        "expected",
        "actual",
        "status",
        "fatal",
    ),
    "date_diagnostics.csv": ("source", "metric", "expected", "actual", "status"),
    "duplicate_multiplicity.csv": ("source", "multiplicity", "key_group_count"),
    "duplicate_summary.csv": (
        "source",
        "key_fields",
        "retained_rows",
        "distinct_keys",
        "unique_permnos",
        "duplicate_key_groups",
        "duplicate_surplus_rows",
        "exact_duplicate_groups",
        "conflicting_duplicate_groups",
        "full_decoded_row_comparison",
    ),
    "interval_diagnostics.csv": ("source", "metric", "expected", "actual", "status"),
    "numeric_validity.csv": ("source", "metric", "expected", "actual", "status"),
    "referential_integrity.csv": (
        "metric",
        "expected",
        "actual",
        "status",
        "fatal",
    ),
    "source_structure.csv": (
        "source",
        "scope_kind",
        "full_file_rows",
        "retained_rows",
        "excluded_rows",
        "header_columns",
        "key_fields",
        "size_bytes",
        "sha256",
        "excluded_non_date_values_compared_or_aggregated",
    ),
}

DIFFERENCE_FIELDS = (
    "check_id",
    "category",
    "expected",
    "actual",
    "evidence",
    "message",
)


class SupportingProfileVerificationError(RuntimeError):
    """Raised when verification cannot establish a safe evidence boundary."""


@dataclasses.dataclass(frozen=True)
class VerificationSettings:
    project_root: Path
    profile_config_path: Path
    paths_config_path: Path
    phase_config_path: Path
    primary_pointer_path: Path
    artifact_root: Path
    database_root: Path
    raw_root: Path
    quality_root: Path
    scratch_root: Path
    config: dict[str, Any]
    block_size_bytes: int = 16 << 20


def canonical_json_text(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def canonical_json_bytes(value: Any) -> bytes:
    return (canonical_json_text(value) + "\n").encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _lexical_absolute(path: str | Path) -> Path:
    return Path(os.path.abspath(os.fspath(Path(path).expanduser())))


def _assert_no_symlink_components(path: Path, label: str) -> None:
    absolute = _lexical_absolute(path)
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(mode):
            raise SupportingProfileVerificationError(
                f"Symlink is prohibited in {label} path: {current}"
            )


def _read_stable(path: Path, label: str) -> bytes:
    path = _lexical_absolute(path)
    _assert_no_symlink_components(path, label)
    try:
        before = path.lstat()
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise SupportingProfileVerificationError(
            f"Cannot open {label}: {path}"
        ) from exc
    try:
        descriptor_before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(descriptor_before.st_mode)
            or before.st_dev != descriptor_before.st_dev
            or before.st_ino != descriptor_before.st_ino
        ):
            raise SupportingProfileVerificationError(
                f"{label} is not the selected regular file"
            )
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1 << 20):
            chunks.append(chunk)
        descriptor_after = os.fstat(descriptor)
        after = path.lstat()
        if (
            descriptor_before.st_dev != descriptor_after.st_dev
            or descriptor_before.st_ino != descriptor_after.st_ino
            or descriptor_before.st_size != descriptor_after.st_size
            or descriptor_before.st_mtime_ns != descriptor_after.st_mtime_ns
            or descriptor_after.st_dev != after.st_dev
            or descriptor_after.st_ino != after.st_ino
            or descriptor_after.st_size != after.st_size
            or descriptor_after.st_mtime_ns != after.st_mtime_ns
        ):
            raise SupportingProfileVerificationError(
                f"{label} changed while being read"
            )
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _read_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    payload = _read_stable(path, label)
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SupportingProfileVerificationError(f"{label} is not UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise SupportingProfileVerificationError(f"{label} is not a JSON object")
    if canonical_json_bytes(value) != payload:
        raise SupportingProfileVerificationError(f"{label} is not canonical JSON")
    return value, payload


def _safe_component(value: Any, label: str) -> str:
    text = str(value)
    if (
        not text
        or text in {".", ".."}
        or Path(text).name != text
        or "/" in text
        or "\\" in text
    ):
        raise SupportingProfileVerificationError(
            f"Unsafe {label} path component: {text!r}"
        )
    return text


def _project_path(settings: VerificationSettings, value: Any, label: str) -> Path:
    text = str(value)
    posix = PurePosixPath(text)
    if (
        posix.is_absolute()
        or not posix.parts
        or any(part in {"", ".", ".."} for part in posix.parts)
        or "\\" in text
    ):
        raise SupportingProfileVerificationError(
            f"Unsafe project-relative {label}: {text!r}"
        )
    result = _lexical_absolute(settings.project_root / Path(*posix.parts))
    try:
        result.relative_to(_lexical_absolute(settings.project_root))
    except ValueError as exc:
        raise SupportingProfileVerificationError(
            f"{label} is outside project root"
        ) from exc
    return result


def _closed_directory(
    path: Path, expected: Sequence[str], label: str
) -> dict[str, bytes]:
    _assert_no_symlink_components(path, label)
    if not path.is_dir():
        raise SupportingProfileVerificationError(f"{label} is not a directory")
    observed: dict[str, bytes] = {}
    for entry in sorted(path.iterdir(), key=lambda item: item.name):
        if entry.is_symlink() or not entry.is_file():
            raise SupportingProfileVerificationError(
                f"{label} contains a non-regular entry: {entry.name}"
            )
        observed[entry.name] = _read_stable(entry, f"{label}/{entry.name}")
    if set(observed) != set(expected) or len(observed) != len(tuple(expected)):
        raise SupportingProfileVerificationError(
            f"{label} closed file set differs: {sorted(observed)}"
        )
    return observed


def _operational_stat(path: Path) -> dict[str, int]:
    metadata = path.stat()
    if not stat.S_ISREG(metadata.st_mode):
        raise SupportingProfileVerificationError(f"Not a regular file: {path}")
    return {
        "device": int(metadata.st_dev),
        "inode": int(metadata.st_ino),
        "mode": int(metadata.st_mode),
        "size_bytes": int(metadata.st_size),
        "mtime_ns": int(metadata.st_mtime_ns),
    }


def load_settings(
    profile_config_path: Path = DEFAULT_PROFILE_CONFIG,
    paths_config_path: Path = DEFAULT_PATHS_CONFIG,
    phase_config_path: Path = DEFAULT_PHASE_CONFIG,
    *,
    primary_pointer_path: Path = DEFAULT_PRIMARY_POINTER,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
    database_root: Path | None = None,
    raw_root: Path | None = None,
    quality_root: Path | None = None,
    scratch_root: Path = DEFAULT_SCRATCH_ROOT,
    block_size_bytes: int = 16 << 20,
) -> VerificationSettings:
    profile_config_path = _lexical_absolute(profile_config_path)
    paths_config_path = _lexical_absolute(paths_config_path)
    phase_config_path = _lexical_absolute(phase_config_path)
    try:
        with profile_config_path.open("rb") as handle:
            config = tomllib.load(handle)
        with paths_config_path.open("rb") as handle:
            paths = tomllib.load(handle)
        with phase_config_path.open("rb") as handle:
            phase = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError, KeyError) as exc:
        raise SupportingProfileVerificationError(
            f"Cannot load verifier settings: {exc}"
        ) from exc
    configured_database = Path(paths["database"]["root"]).expanduser()
    selected_database = _lexical_absolute(
        database_root or os.environ.get("PORTFOLIO_DATABASE_ROOT", configured_database)
    )
    namespace = phase["storage"]["quality_namespace"]
    return VerificationSettings(
        project_root=PROJECT_ROOT,
        profile_config_path=profile_config_path,
        paths_config_path=paths_config_path,
        phase_config_path=phase_config_path,
        primary_pointer_path=_lexical_absolute(primary_pointer_path),
        artifact_root=_lexical_absolute(artifact_root),
        database_root=selected_database,
        raw_root=_lexical_absolute(
            raw_root or selected_database / paths["database"]["raw"]
        ),
        quality_root=_lexical_absolute(
            quality_root or selected_database / paths["database"]["quality"] / namespace
        ),
        scratch_root=_lexical_absolute(scratch_root),
        config=config,
        block_size_bytes=int(block_size_bytes),
    )


class _Checks:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def add(
        self,
        check_id: str,
        category: str,
        *,
        expected: Any,
        actual: Any,
        passed: bool,
        evidence: str,
        message: str,
    ) -> None:
        self.rows.append(
            {
                "check_id": check_id,
                "category": category,
                "expected": expected,
                "actual": actual,
                "status": "PASS" if passed else "FAIL",
                "evidence": evidence,
                "message": message,
            }
        )

    def sorted(self) -> list[dict[str, Any]]:
        return sorted(self.rows, key=lambda row: row["check_id"].encode("utf-8"))


def _authenticate_hash(
    path: Path,
    expected: str,
    label: str,
    snapshots: dict[Path, str],
) -> bytes:
    if SHA256_RE.fullmatch(str(expected)) is None:
        raise SupportingProfileVerificationError(f"Invalid locked SHA-256 for {label}")
    payload = _read_stable(path, label)
    actual = sha256_bytes(payload)
    if actual != expected:
        raise SupportingProfileVerificationError(
            f"{label} SHA-256 differs: expected {expected}, actual {actual}"
        )
    snapshots[_lexical_absolute(path)] = actual
    return payload


def _authenticate_locked_context(
    settings: VerificationSettings,
    binding: Mapping[str, Any],
    snapshots: dict[Path, str],
) -> None:
    config = settings.config
    profile_payload = _authenticate_hash(
        settings.profile_config_path,
        binding["profile_config_sha256"],
        "supporting profile config",
        snapshots,
    )
    try:
        if tomllib.loads(profile_payload.decode("utf-8")) != config:
            raise SupportingProfileVerificationError(
                "Loaded profile config differs from authenticated bytes"
            )
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise SupportingProfileVerificationError(
            "Authenticated profile config is invalid"
        ) from exc
    _authenticate_hash(
        settings.paths_config_path,
        binding["paths_config_sha256"],
        "paths config",
        snapshots,
    )
    phase = config["upstream"]["phase_02"]
    expected_phase_path = _project_path(settings, phase["phase_config"], "phase config")
    if expected_phase_path != settings.phase_config_path:
        raise SupportingProfileVerificationError("Selected phase config path differs")
    _authenticate_hash(
        expected_phase_path,
        phase["phase_config_sha256"],
        "phase config",
        snapshots,
    )
    if binding["phase_config_sha256"] != phase["phase_config_sha256"]:
        raise SupportingProfileVerificationError("Manifest phase binding differs")

    methodology = config["upstream"]["methodology"]
    methodology_payload = _authenticate_hash(
        _project_path(settings, methodology["freeze_manifest"], "methodology freeze"),
        methodology["freeze_manifest_sha256"],
        "methodology freeze",
        snapshots,
    )
    methodology_object = json.loads(methodology_payload.decode("utf-8"))
    if methodology_object.get("status") != "frozen_preimplementation":
        raise SupportingProfileVerificationError("Methodology is not frozen")
    if binding["methodology_freeze_sha256"] != methodology["freeze_manifest_sha256"]:
        raise SupportingProfileVerificationError("Manifest methodology binding differs")

    phase_01 = config["upstream"]["phase_01"]
    freeze_payload = _authenticate_hash(
        _project_path(settings, phase_01["freeze_manifest"], "Phase 1 freeze"),
        phase_01["freeze_manifest_sha256"],
        "Phase 1 freeze",
        snapshots,
    )
    freeze_object = json.loads(freeze_payload.decode("utf-8"))
    if (
        freeze_object.get("freeze_id") != phase_01["freeze_id"]
        or freeze_object.get("status") != "phase_01_frozen"
    ):
        raise SupportingProfileVerificationError("Phase 1 freeze identity differs")
    if binding["phase_01_freeze_manifest_sha256"] != phase_01["freeze_manifest_sha256"]:
        raise SupportingProfileVerificationError("Manifest Phase 1 binding differs")

    for kind in ("contract", "calendar", "daily_monthly_profile"):
        locked = config["upstream"][kind]
        pointer_path = _project_path(
            settings, locked["verified_pointer"], f"{kind} pointer"
        )
        payload = _authenticate_hash(
            pointer_path,
            locked["verified_pointer_sha256"],
            f"{kind} pointer",
            snapshots,
        )
        pointer = json.loads(payload.decode("utf-8"))
        expected_id = "contract_id" if kind == "contract" else "build_id"
        if (
            pointer.get("status") != locked["required_status"]
            or pointer.get(expected_id) != locked[expected_id]
            or pointer.get("verification_id") != locked["verification_id"]
        ):
            raise SupportingProfileVerificationError(
                f"Authenticated {kind} pointer identity differs"
            )
        binding_id = (
            "contract_id"
            if kind == "contract"
            else (
                "calendar_build_id"
                if kind == "calendar"
                else "daily_monthly_profile_build_id"
            )
        )
        binding_verification = (
            "contract_verification_id"
            if kind == "contract"
            else (
                "calendar_verification_id"
                if kind == "calendar"
                else "daily_monthly_profile_verification_id"
            )
        )
        if (
            binding[binding_id] != locked[expected_id]
            or binding[binding_verification] != locked["verification_id"]
        ):
            raise SupportingProfileVerificationError(f"Manifest {kind} binding differs")


def _parse_inventory(payload: bytes) -> dict[str, str]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SupportingProfileVerificationError("Hash inventory is not UTF-8") from exc
    result: dict[str, str] = {}
    if text and not text.endswith("\n"):
        raise SupportingProfileVerificationError("Hash inventory lacks trailing LF")
    for line in text.splitlines():
        parts = line.split("  ", 1)
        if len(parts) != 2 or SHA256_RE.fullmatch(parts[0]) is None or not parts[1]:
            raise SupportingProfileVerificationError("Malformed hash inventory")
        if parts[1] in result:
            raise SupportingProfileVerificationError("Duplicate inventory path")
        result[parts[1]] = parts[0]
    if list(result) != sorted(result):
        raise SupportingProfileVerificationError("Hash inventory is not sorted")
    return result


def _read_primary_csv(payload: bytes, filename: str) -> list[dict[str, str]]:
    try:
        reader = csv.DictReader(io.StringIO(payload.decode("utf-8"), newline=""))
        if tuple(reader.fieldnames or ()) != CSV_SCHEMAS[filename]:
            raise SupportingProfileVerificationError(
                f"Primary {filename} header differs"
            )
        rows = list(reader)
    except (UnicodeDecodeError, csv.Error) as exc:
        raise SupportingProfileVerificationError(
            f"Cannot decode primary {filename}"
        ) from exc
    if any(None in row for row in rows):
        raise SupportingProfileVerificationError(
            f"Primary {filename} contains surplus fields"
        )
    return rows


def authenticate_primary_bundle(
    settings: VerificationSettings, checks: _Checks
) -> dict[str, Any]:
    snapshots: dict[Path, str] = {}
    pointer, pointer_payload = _read_json(
        settings.primary_pointer_path, "primary pointer"
    )
    snapshots[settings.primary_pointer_path] = sha256_bytes(pointer_payload)
    expected_pointer_keys = {
        "artifact_path",
        "build_id",
        "contract_id",
        "implementation",
        "manifest_sha256",
        "schema_version",
        "status",
    }
    if set(pointer) != expected_pointer_keys or pointer.get("status") != PRIMARY_STATUS:
        raise SupportingProfileVerificationError(
            "Primary pointer schema or status differs"
        )
    build_id = _safe_component(pointer["build_id"], "build_id")
    contract_id = _safe_component(pointer["contract_id"], "contract_id")
    artifact_subdirectory = _safe_component(
        settings.config["outputs"]["artifact_subdirectory"],
        "artifact subdirectory",
    )
    artifact_directory = _lexical_absolute(
        settings.artifact_root / artifact_subdirectory / build_id
    )
    if (
        _project_path(settings, pointer["artifact_path"], "artifact path")
        != artifact_directory
    ):
        raise SupportingProfileVerificationError(
            "Primary pointer artifact path differs from locked destination"
        )
    artifact_payloads = _closed_directory(
        artifact_directory,
        settings.config["outputs"]["closed_artifact_file_set"],
        "primary artifact",
    )
    for name, payload in artifact_payloads.items():
        snapshots[artifact_directory / name] = sha256_bytes(payload)
    manifest_payload = artifact_payloads["manifest.json"]
    if sha256_bytes(manifest_payload) != pointer["manifest_sha256"]:
        raise SupportingProfileVerificationError("Pointer manifest hash differs")
    try:
        manifest = json.loads(manifest_payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SupportingProfileVerificationError("Manifest is not JSON") from exc
    if canonical_json_bytes(manifest) != manifest_payload or set(manifest) != {
        "binding",
        "build_id",
        "contract_id",
        "outputs",
        "profile_name",
        "schema_version",
        "status",
        "summary",
    }:
        raise SupportingProfileVerificationError(
            "Manifest schema/serialization differs"
        )
    if any(
        (
            manifest.get("build_id") != build_id,
            manifest.get("contract_id") != contract_id,
            manifest.get("status") != PRIMARY_STATUS,
            manifest.get("profile_name") != settings.config["profile_name"],
        )
    ):
        raise SupportingProfileVerificationError("Manifest identity differs")
    binding = manifest.get("binding")
    if not isinstance(binding, dict):
        raise SupportingProfileVerificationError("Manifest binding is not an object")
    _authenticate_locked_context(settings, binding, snapshots)

    implementation = pointer.get("implementation")
    expected_implementation_keys = {
        "cli_sha256",
        "compiler_sha256",
        "publisher_sha256",
    }
    if (
        not isinstance(implementation, dict)
        or set(implementation) != expected_implementation_keys
    ):
        raise SupportingProfileVerificationError(
            "Primary implementation binding differs"
        )
    primary_implementations = {
        "compiler_sha256": settings.project_root
        / "src/portfolio_bot/data/phase_02_crsp_supporting_profile.py",
        "publisher_sha256": settings.project_root
        / "src/portfolio_bot/data/phase_02_crsp_supporting_profile_publication.py",
        "cli_sha256": settings.project_root
        / "scripts/build_phase_02_crsp_supporting_profile.py",
    }
    for key, path in primary_implementations.items():
        payload = _read_stable(path, f"primary {key}")
        digest = sha256_bytes(payload)
        snapshots[path] = digest
        if implementation[key] != digest:
            raise SupportingProfileVerificationError(
                f"Primary {key} binding differs from current bytes"
            )
    if binding["implementation_sha256"] != implementation["compiler_sha256"]:
        raise SupportingProfileVerificationError(
            "Manifest compiler binding differs from pointer"
        )

    quality_subdirectory = _safe_component(
        settings.config["outputs"]["quality_subdirectory"],
        "quality subdirectory",
    )
    quality_directory = _lexical_absolute(
        settings.quality_root / contract_id / quality_subdirectory / build_id
    )
    quality_payloads = _closed_directory(
        quality_directory,
        settings.config["outputs"]["closed_quality_file_set"],
        "primary quality",
    )
    for name, payload in quality_payloads.items():
        snapshots[quality_directory / name] = sha256_bytes(payload)
    expected_outputs = [
        {
            "path": (
                quality_directory.relative_to(settings.database_root) / name
            ).as_posix(),
            "sha256": sha256_bytes(quality_payloads[name]),
            "size_bytes": len(quality_payloads[name]),
        }
        for name in sorted(quality_payloads)
    ]
    if manifest.get("outputs") != expected_outputs:
        raise SupportingProfileVerificationError(
            "Manifest outputs differ from quality bytes"
        )
    expected_inventory = {item["path"]: item["sha256"] for item in expected_outputs}
    if (
        _parse_inventory(artifact_payloads["output_hashes.sha256"])
        != expected_inventory
    ):
        raise SupportingProfileVerificationError("Output inventory differs")

    json_payloads: dict[str, dict[str, Any]] = {}
    csv_payloads: dict[str, list[dict[str, str]]] = {}
    for name, payload in quality_payloads.items():
        if name.endswith(".json"):
            try:
                value = json.loads(payload.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise SupportingProfileVerificationError(
                    f"Primary {name} is not JSON"
                ) from exc
            if not isinstance(value, dict) or canonical_json_bytes(value) != payload:
                raise SupportingProfileVerificationError(
                    f"Primary {name} is not canonical object JSON"
                )
            json_payloads[name] = value
        elif name in CSV_SCHEMAS:
            csv_payloads[name] = _read_primary_csv(payload, name)
        else:
            raise SupportingProfileVerificationError(
                f"Unrecognized primary quality file: {name}"
            )
    checks.add(
        "PRIMARY_BUNDLE_AUTHENTICATED",
        "primary_bundle",
        expected=True,
        actual=True,
        passed=True,
        evidence=f"{len(quality_payloads)} quality + {len(artifact_payloads)} artifact files",
        message="Primary pointer, manifest, inventory, closed sets, and hashes authenticate.",
    )
    return {
        "artifact_directory": artifact_directory,
        "binding": binding,
        "csv": csv_payloads,
        "json": json_payloads,
        "manifest": manifest,
        "manifest_path": artifact_directory / "manifest.json",
        "manifest_sha256": sha256_bytes(manifest_payload),
        "pointer": pointer,
        "quality_directory": quality_directory,
        "quality_payloads": quality_payloads,
        "snapshot": snapshots,
    }


def _calendar_dates(
    settings: VerificationSettings, snapshots: dict[Path, str]
) -> tuple[list[dt.date], dict[str, Any]]:
    locked = settings.config["upstream"]["calendar"]
    pointer_path = _project_path(
        settings, locked["verified_pointer"], "calendar pointer"
    )
    pointer_payload = _authenticate_hash(
        pointer_path,
        locked["verified_pointer_sha256"],
        "calendar pointer",
        snapshots,
    )
    pointer = json.loads(pointer_payload.decode("utf-8"))
    manifest_path = (
        _project_path(settings, pointer["primary_artifact_path"], "calendar artifact")
        / "manifest.json"
    )
    manifest_payload = _authenticate_hash(
        manifest_path,
        pointer["primary_manifest_sha256"],
        "calendar manifest",
        snapshots,
    )
    manifest = json.loads(manifest_payload.decode("utf-8"))
    candidates = [
        item
        for item in manifest.get("outputs", [])
        if Path(str(item.get("path", ""))).name == "trading_calendar.csv"
    ]
    if len(candidates) != 1:
        raise SupportingProfileVerificationError(
            "Calendar manifest lacks one trading_calendar.csv"
        )
    output = candidates[0]
    calendar_path = _lexical_absolute(output["path"])
    calendar_payload = _authenticate_hash(
        calendar_path, output["sha256"], "trading calendar", snapshots
    )
    try:
        reader = csv.DictReader(
            io.StringIO(calendar_payload.decode("utf-8"), newline="")
        )
        fields = reader.fieldnames or []
        candidates = [
            name
            for name in fields
            if name in {"calendar_date", "date", "observed_date", "trading_date"}
        ]
        if len(candidates) != 1:
            raise SupportingProfileVerificationError(
                "Calendar date column is ambiguous"
            )
        dates = [dt.date.fromisoformat(row[candidates[0]]) for row in reader]
    except (UnicodeDecodeError, csv.Error, ValueError) as exc:
        raise SupportingProfileVerificationError(
            "Cannot decode trading calendar"
        ) from exc
    if len(set(dates)) != len(dates):
        raise SupportingProfileVerificationError("Trading calendar contains duplicates")
    try:
        portable_path = calendar_path.relative_to(settings.database_root).as_posix()
    except ValueError as exc:
        raise SupportingProfileVerificationError(
            "Trading calendar is outside database root"
        ) from exc
    return dates, {
        "path": portable_path,
        "sha256": output["sha256"],
        "size_bytes": int(output["size_bytes"]),
        "date_count": len(dates),
    }


def _wire(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list, tuple)):
        return canonical_json_text(value)
    return str(value)


def _csv_bytes(fields: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output,
        fieldnames=list(fields),
        delimiter=",",
        quotechar='"',
        doublequote=True,
        escapechar=None,
        quoting=csv.QUOTE_MINIMAL,
        lineterminator="\n",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow({field: _wire(row.get(field)) for field in fields})
    return output.getvalue().encode("utf-8")


def _actual_metric(result: Mapping[str, Any], metric: str) -> int:
    if metric in result:
        return int(result[metric])
    duplicate_mapping = {
        "distinct_keys": result["duplicates"]["distinct_keys"],
        "duplicate_key_groups": result["duplicates"]["duplicate_key_groups"],
        "duplicate_surplus_rows": result["duplicates"]["duplicate_surplus_rows"],
        "conflicting_duplicate_groups": result["duplicates"][
            "conflicting_duplicate_groups"
        ],
    }
    if metric in duplicate_mapping:
        return int(duplicate_mapping[metric])
    for group in ("date_metrics", "numeric_metrics", "interval_metrics"):
        if metric in result[group]:
            return int(result[group][metric])
    return 0


def _metric_category(metric: str, source: Mapping[str, Any]) -> str:
    if metric.startswith("blank_") and metric.removeprefix("blank_") in source.get(
        "positive_integer_columns", []
    ):
        return "numeric"
    if metric.startswith(("duplicate_", "distinct_")):
        return "duplicates"
    if metric.startswith(
        (
            "blank_",
            "declare_",
            "record_",
            "pay_",
            "posting_",
            "next_",
            "amount_",
            "delisting_",
            "both_",
            "security_",
            "information_",
        )
    ):
        return "dates"
    if metric.startswith(("invalid_", "nonpositive_")):
        return "numeric"
    if metric in {
        "inverted_intervals",
        "overlap_edges",
        "contiguous_edges",
        "gap_edges",
        "starts_before_scope",
        "ends_after_scope",
        "spans_entire_scope",
    }:
        return "intervals"
    return "structure"


def _primary_check_row(
    check_id: str,
    source: str,
    category: str,
    metric: str,
    expected: Any,
    actual: Any,
    *,
    fatal: bool = True,
) -> dict[str, Any]:
    return {
        "check_id": check_id,
        "source": source,
        "category": category,
        "metric": metric,
        "expected": expected,
        "actual": actual,
        "status": "PASS" if expected == actual else "FAIL",
        "fatal": fatal,
    }


def _referential_check_is_fatal(metric: str) -> bool:
    """Reconstruct the locked fatal/diagnostic referential policy.

    The zero- and multiple-match metric names use the plural ``matches`` while
    the exactly-one metric uses singular ``match``.  Classify the policy from
    the stable date suffix so all three variants receive the same treatment.
    """

    return (
        metric
        in {
            "verified_calendar_dates",
            "names_invalid_or_nonpositive_permco_rows",
            "names_nonconstant_security_lifetime_permnos",
            "names_permnos_missing_current_header_key",
        }
        or metric.startswith("shares_rows_with_")
        or (
            metric.startswith("distribution_rows_with_")
            and metric.endswith("_on_ex_date")
        )
        or (
            metric.startswith("delisting_rows_with_")
            and metric.endswith("_on_delisting_date")
        )
    )


def reconstruct_aggregate_rows(
    config: Mapping[str, Any], scan: Mapping[str, Any]
) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {
        "source_structure": [],
        "duplicate_summary": [],
        "duplicate_multiplicity": [],
        "categorical_domains": [],
        "date_diagnostics": [],
        "interval_diagnostics": [],
        "numeric_validity": [],
        "referential_integrity": [],
        "checks": [],
    }
    results = {row["source"]: row for row in scan["sources"]}
    sources = {row["filename"]: row for row in config["sources"]}
    for metric, wanted in config["referential_integrity"].items():
        if isinstance(wanted, bool) or not isinstance(wanted, int):
            continue
        actual = int(scan["referential"].get(metric, 0))
        fatal = _referential_check_is_fatal(metric)
        row = {
            "metric": metric,
            "expected": wanted,
            "actual": actual,
            "status": "PASS" if actual == wanted else "FAIL",
            "fatal": fatal,
        }
        output["referential_integrity"].append(row)
        output["checks"].append(
            _primary_check_row(
                f"REFERENTIAL::{metric}",
                "ALL",
                "referential_integrity",
                metric,
                wanted,
                actual,
                fatal=fatal,
            )
        )
    for filename in SOURCE_ORDER:
        result = results[filename]
        source = sources[filename]
        receipt = scan["receipts"][filename]
        output["source_structure"].append(
            {
                "source": filename,
                "scope_kind": result["scope_kind"],
                "full_file_rows": result["full_file_rows"],
                "retained_rows": result["retained_rows"],
                "excluded_rows": result["excluded_rows"],
                "header_columns": result["header_columns"],
                "key_fields": result["key_fields"],
                "size_bytes": receipt["size_bytes"],
                "sha256": receipt["sha256"],
                "excluded_non_date_values_compared_or_aggregated": result[
                    "excluded_non_date_values_compared_or_aggregated"
                ],
            }
        )
        output["duplicate_summary"].append(
            {
                "source": filename,
                "key_fields": result["key_fields"],
                "retained_rows": result["retained_rows"],
                "distinct_keys": result["duplicates"]["distinct_keys"],
                "unique_permnos": result["unique_permnos"],
                "duplicate_key_groups": result["duplicates"]["duplicate_key_groups"],
                "duplicate_surplus_rows": result["duplicates"][
                    "duplicate_surplus_rows"
                ],
                "exact_duplicate_groups": result["duplicates"][
                    "exact_duplicate_groups"
                ],
                "conflicting_duplicate_groups": result["duplicates"][
                    "conflicting_duplicate_groups"
                ],
                "full_decoded_row_comparison": source["scope_kind"] != "qa_key_only",
            }
        )
        for multiplicity, groups in result["duplicates"]["multiplicity"].items():
            output["duplicate_multiplicity"].append(
                {
                    "source": filename,
                    "multiplicity": multiplicity,
                    "key_group_count": groups,
                }
            )
        expected = config["expected"][filename.replace(".", "_")]
        for metric, wanted in expected.items():
            if isinstance(wanted, bool) or not isinstance(wanted, int):
                continue
            actual = _actual_metric(result, metric)
            category = _metric_category(metric, source)
            output["checks"].append(
                _primary_check_row(
                    f"SOURCE::{filename}::{metric}",
                    filename,
                    category,
                    metric,
                    wanted,
                    actual,
                )
            )
            diagnostic = {
                "source": filename,
                "metric": metric,
                "expected": wanted,
                "actual": actual,
                "status": "PASS" if wanted == actual else "FAIL",
            }
            if category == "dates":
                output["date_diagnostics"].append(diagnostic)
            elif category == "intervals":
                output["interval_diagnostics"].append(diagnostic)
            elif category == "numeric":
                output["numeric_validity"].append(diagnostic)
        domains = config.get("expected_domains", {}).get(filename.replace(".", "_"), {})
        for column in source.get("categorical_columns", []):
            counts = result["categorical_counts"][column]
            expected_values = list(domains[column])
            observed = sorted(
                (value for value in counts if value != ""),
                key=lambda value: value.encode("utf-8"),
            )
            output["checks"].append(
                _primary_check_row(
                    f"DOMAIN::{filename}::{column}",
                    filename,
                    "categorical_domain",
                    column,
                    expected_values,
                    observed,
                )
            )
            output["checks"].append(
                _primary_check_row(
                    f"FREQUENCY::{filename}::{column}",
                    filename,
                    "categorical_frequency",
                    column,
                    result["retained_rows"],
                    sum(counts.values()),
                )
            )
            for raw_value, count in counts.items():
                output["categorical_domains"].append(
                    {
                        "source": filename,
                        "column": column,
                        "raw_value": raw_value,
                        "row_count": count,
                        "is_blank": raw_value == "",
                        "classification": (
                            "blank"
                            if raw_value == ""
                            else (
                                "expected_nonblank"
                                if raw_value in expected_values
                                else "unexpected_nonblank"
                            )
                        ),
                    }
                )
    order = {name: position for position, name in enumerate(SOURCE_ORDER)}
    for name, rows in output.items():
        if name == "checks":
            rows.sort(key=lambda row: row["check_id"].encode("utf-8"))
        elif name == "referential_integrity":
            rows.sort(key=lambda row: row["metric"].encode("utf-8"))
        elif name == "categorical_domains":
            rows.sort(
                key=lambda row: (
                    order[row["source"]],
                    row["column"].encode("utf-8"),
                    row["raw_value"] != "",
                    row["raw_value"].encode("utf-8"),
                )
            )
        else:
            rows.sort(
                key=lambda row: (
                    order.get(row.get("source", SOURCE_ORDER[0]), -1),
                    str(row.get("metric", row.get("multiplicity", ""))).encode("utf-8"),
                )
            )
    return output


def _build_id(
    binding: Mapping[str, Any],
    rows: Mapping[str, Any],
    receipts: Mapping[str, Any],
) -> str:
    receipt_fields = (
        "filename",
        "sha256",
        "size_bytes",
        "full_file_rows_expected",
        "full_file_rows_scanned",
    )
    portable_receipts = {
        name: {field: receipt[field] for field in receipt_fields if field in receipt}
        for name, receipt in sorted(receipts.items())
    }
    identity = {
        "binding": binding,
        "quality_rows": rows,
        "source_receipts": portable_receipts,
    }
    return f"crsp-support-{sha256_bytes(canonical_json_bytes(identity))[:24]}"


def compare_reconstruction(
    settings: VerificationSettings,
    primary: Mapping[str, Any],
    scan: Mapping[str, Any],
    calendar_receipt: Mapping[str, Any],
    checks: _Checks,
) -> dict[str, Any]:
    rows = reconstruct_aggregate_rows(settings.config, scan)
    rendered_csv = {
        filename: _csv_bytes(CSV_SCHEMAS[filename], rows[filename.removesuffix(".csv")])
        for filename in CSV_SCHEMAS
    }
    for filename, expected_payload in rendered_csv.items():
        actual_payload = primary["quality_payloads"][filename]
        checks.add(
            f"SEMANTIC_BYTES::{filename}",
            "semantic_reconstruction",
            expected=sha256_bytes(expected_payload),
            actual=sha256_bytes(actual_payload),
            passed=expected_payload == actual_payload,
            evidence=f"aggregate rows={len(rows[filename.removesuffix('.csv')])}",
            message=f"Independent aggregate reconstruction for {filename}.",
        )
    binding = primary["binding"]
    reconstructed_build_id = _build_id(binding, rows, scan["receipts"])
    checks.add(
        "BUILD_ID_RECONSTRUCTED",
        "content_identity",
        expected=reconstructed_build_id,
        actual=primary["pointer"]["build_id"],
        passed=reconstructed_build_id == primary["pointer"]["build_id"],
        evidence="binding + aggregate rows + stable source receipts",
        message="Primary build ID is independently content-bound.",
    )
    build_id = primary["pointer"]["build_id"]
    input_receipt = {
        "schema_version": 1,
        "build_id": build_id,
        "binding": binding,
        "calendar": dict(calendar_receipt),
        "sources": [scan["receipts"][filename] for filename in SOURCE_ORDER],
    }
    contract = {
        "schema_version": 1,
        "profile_name": settings.config["profile_name"],
        "profile_version": settings.config["profile_version"],
        "scope": settings.config["scope"],
        "key_semantics": settings.config["key_semantics"],
        "duplicate_policy": settings.config["duplicate_policy"],
        "interval_policy": settings.config["interval_policy"],
        "categorical_policy": settings.config["categorical_policy"],
        "diagnostic_status": settings.config["diagnostic_status"],
        "csv_schemas": {name: list(fields) for name, fields in CSV_SCHEMAS.items()},
        "closed_quality_file_set": settings.config["outputs"][
            "closed_quality_file_set"
        ],
    }
    primary_checks = rows["checks"]
    summary = {
        "schema_version": 1,
        "profile_name": settings.config["profile_name"],
        "build_id": build_id,
        "status": PRIMARY_STATUS,
        "source_count": scan["summary"]["source_count"],
        "full_file_rows_scanned": scan["summary"]["full_file_rows_read"],
        "retained_rows": scan["summary"]["retained_rows"],
        "check_count": len(primary_checks),
        "pass_count": sum(row["status"] == "PASS" for row in primary_checks),
        "fail_count": sum(row["status"] != "PASS" for row in primary_checks),
        "canonical_tables_materialized": False,
        "numeric_units_validated": False,
        "returns_reconciled": False,
        "features_built": False,
        "targets_built": False,
        "portfolios_built": False,
        "pnl_built": False,
        "performance_analyzed": False,
    }
    expected_json = {
        "input_receipt.json": input_receipt,
        "profile_contract.json": contract,
        "summary.json": summary,
        "_SUCCESS.json": summary,
    }
    for filename, expected in expected_json.items():
        actual = primary["json"][filename]
        checks.add(
            f"SEMANTIC_JSON::{filename}",
            "semantic_reconstruction",
            expected=sha256_bytes(canonical_json_bytes(expected)),
            actual=sha256_bytes(canonical_json_bytes(actual)),
            passed=actual == expected,
            evidence=filename,
            message=f"Independent deterministic reconstruction for {filename}.",
        )
    for source in settings.config["sources"]:
        filename = source["filename"]
        observed = scan["receipts"][filename]
        checks.add(
            f"RAW_SHA256::{filename}",
            "raw_authentication",
            expected=source["sha256"],
            actual=observed["sha256"],
            passed=source["sha256"] == observed["sha256"],
            evidence=f"size={observed['size_bytes']}",
            message="Independent full-file raw SHA-256 recomputation.",
        )
        checks.add(
            f"RAW_ROWS::{filename}",
            "raw_authentication",
            expected=int(source["full_file_rows"]),
            actual=int(observed["full_file_rows_scanned"]),
            passed=int(source["full_file_rows"])
            == int(observed["full_file_rows_scanned"]),
            evidence=filename,
            message="Independent full-file CSV row count.",
        )
    independence = scan["independence"]
    expected_independence = {
        "aggregate_only": True,
        "full_raw_hashes_recomputed": True,
        "grouping_depends_on_input_sort": False,
        "primary_compiler_imported": False,
        "primary_publisher_imported": False,
        "raw_files_reread": 5,
    }
    checks.add(
        "INDEPENDENCE_CONTRACT",
        "independence",
        expected=expected_independence,
        actual=independence,
        passed=independence == expected_independence,
        evidence="standalone scanner result",
        message="Independent scan is aggregate-only and order-independent.",
    )
    return {
        "build_id": reconstructed_build_id,
        "input_receipt": input_receipt,
        "profile_contract": contract,
        "rows": rows,
        "summary": summary,
    }


def verifier_implementation_paths(
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Path]:
    labels = (
        "scripts/verify_phase_02_crsp_supporting_profile.py",
        "src/portfolio_bot/data/phase_02_crsp_supporting_profile_independent_scan.py",
        "src/portfolio_bot/data/phase_02_crsp_supporting_profile_verification_publication.py",
        "src/portfolio_bot/data/verify_phase_02_crsp_supporting_profile.py",
    )
    return {label: _lexical_absolute(project_root / label) for label in labels}


def build_independent_report(
    settings: VerificationSettings,
    primary: Mapping[str, Any],
    scan: Mapping[str, Any],
    checks: _Checks,
    implementation_sha256: str,
) -> dict[str, Any]:
    from portfolio_bot.data.phase_02_crsp_supporting_profile_verification_publication import (
        bind_verification_id,
        render_differences_csv,
    )

    rows = checks.sorted()
    fail_count = sum(row["status"] != "PASS" for row in rows)
    report = {
        "build_id": primary["pointer"]["build_id"],
        "checks": rows,
        "contract_id": primary["pointer"]["contract_id"],
        "independence": scan["independence"],
        "primary_manifest_sha256": primary["manifest_sha256"],
        "profile_name": settings.config["profile_name"],
        "schema_version": int(settings.config["schema_version"]),
        "status": PASS_STATUS if fail_count == 0 else FAIL_STATUS,
        "summary": {
            "check_count": len(rows),
            "difference_count": fail_count,
            "fail_count": fail_count,
            "full_file_rows_read": scan["summary"]["full_file_rows_read"],
            "pass_count": len(rows) - fail_count,
            "retained_rows": scan["summary"]["retained_rows"],
            "source_count": scan["summary"]["source_count"],
        },
        "verification_id": "",
        "verifier_implementation_sha256": implementation_sha256,
    }
    return bind_verification_id(report, render_differences_csv(report))


def _scan_call(
    settings: VerificationSettings,
    calendar_dates: Sequence[dt.date],
    scratch_directory: Path,
) -> Mapping[str, Any]:
    from portfolio_bot.data.phase_02_crsp_supporting_profile_independent_scan import (
        scan_supporting_sources_independently,
    )

    return scan_supporting_sources_independently(
        settings.config,
        {filename: settings.raw_root / filename for filename in SOURCE_ORDER},
        calendar_dates=calendar_dates,
        scratch_directory=scratch_directory,
        block_size_bytes=settings.block_size_bytes,
    )


def independently_verify_supporting_profile(
    settings: VerificationSettings | None = None,
    *,
    scan_result: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    settings = settings or load_settings()
    checks = _Checks()
    primary = authenticate_primary_bundle(settings, checks)
    snapshots = dict(primary["snapshot"])
    calendar_dates, calendar_receipt = _calendar_dates(settings, snapshots)
    from portfolio_bot.data.phase_02_crsp_supporting_profile_verification_publication import (
        implementation_bundle_sha256,
        render_differences_csv,
    )

    implementation_paths = verifier_implementation_paths(settings.project_root)
    implementation_before = implementation_bundle_sha256(
        implementation_paths, project_root=settings.project_root
    )
    if scan_result is None:
        _assert_no_symlink_components(settings.scratch_root, "verifier scratch")
        settings.scratch_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        _assert_no_symlink_components(settings.scratch_root, "verifier scratch")
        with tempfile.TemporaryDirectory(
            prefix="crsp-supporting-verify-", dir=settings.scratch_root
        ) as temporary:
            scan = _scan_call(settings, calendar_dates, Path(temporary))
    else:
        scan = dict(scan_result)
        checks.add(
            "INJECTED_SCAN_RESULT_PROHIBITED",
            "independence",
            expected=False,
            actual=True,
            passed=False,
            evidence="in-memory test hook",
            message="Injected scans cannot produce a verified publication.",
        )
    reconstructed = compare_reconstruction(
        settings, primary, scan, calendar_receipt, checks
    )
    implementation_after = implementation_bundle_sha256(
        implementation_paths, project_root=settings.project_root
    )
    if implementation_after != implementation_before:
        raise SupportingProfileVerificationError(
            "Verifier implementation changed during independent scan"
        )
    report = build_independent_report(
        settings, primary, scan, checks, implementation_before
    )
    differences = render_differences_csv(report)
    raw_snapshot = {
        _lexical_absolute(settings.raw_root / filename): scan["raw_operational"][
            filename
        ]["after"]
        for filename in SOURCE_ORDER
    }
    return {
        "differences_payload": differences,
        "final_snapshot": snapshots,
        "implementation_paths": implementation_paths,
        "primary": primary,
        "raw_operational_snapshot": raw_snapshot,
        "reconstructed": reconstructed,
        "report": report,
        "scan": scan,
    }


def verify_and_publish_supporting_profile(
    settings: VerificationSettings | None = None,
    *,
    scan_result: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if scan_result is not None:
        raise SupportingProfileVerificationError(
            "Injected scan results are prohibited from publication"
        )
    settings = settings or load_settings()
    result = independently_verify_supporting_profile(settings)
    from portfolio_bot.data.phase_02_crsp_supporting_profile_verification_publication import (
        publish_independent_verification,
    )

    published = publish_independent_verification(
        result["report"],
        result["differences_payload"],
        artifact_root=settings.artifact_root,
        project_root=settings.project_root,
        config=settings.config,
        verifier_implementation_paths=result["implementation_paths"],
        final_snapshot=result["final_snapshot"],
        raw_operational_snapshot=result["raw_operational_snapshot"],
        primary_manifest_path=result["primary"]["manifest_path"],
    )
    return {**result, "published": published}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-config", type=Path, default=DEFAULT_PROFILE_CONFIG)
    parser.add_argument("--paths-config", type=Path, default=DEFAULT_PATHS_CONFIG)
    parser.add_argument("--phase-config", type=Path, default=DEFAULT_PHASE_CONFIG)
    parser.add_argument("--primary-pointer", type=Path, default=DEFAULT_PRIMARY_POINTER)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--database-root", type=Path)
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--quality-root", type=Path)
    parser.add_argument("--scratch-root", type=Path, default=DEFAULT_SCRATCH_ROOT)
    parser.add_argument("--block-size-bytes", type=int, default=16 << 20)
    parser.add_argument("--no-publish", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    settings = load_settings(
        arguments.profile_config,
        arguments.paths_config,
        arguments.phase_config,
        primary_pointer_path=arguments.primary_pointer,
        artifact_root=arguments.artifact_root,
        database_root=arguments.database_root,
        raw_root=arguments.raw_root,
        quality_root=arguments.quality_root,
        scratch_root=arguments.scratch_root,
        block_size_bytes=arguments.block_size_bytes,
    )
    result = (
        independently_verify_supporting_profile(settings)
        if arguments.no_publish
        else verify_and_publish_supporting_profile(settings)
    )
    report = result["report"]
    print(
        canonical_json_text(
            {
                "build_id": report["build_id"],
                "difference_count": report["summary"]["difference_count"],
                "status": report["status"],
                "verification_id": report["verification_id"],
            }
        )
    )
    return 0 if report["status"] == PASS_STATUS else 1


__all__ = [
    "SupportingProfileVerificationError",
    "VerificationSettings",
    "authenticate_primary_bundle",
    "build_independent_report",
    "canonical_json_bytes",
    "canonical_json_text",
    "compare_reconstruction",
    "independently_verify_supporting_profile",
    "load_settings",
    "main",
    "reconstruct_aggregate_rows",
    "verifier_implementation_paths",
    "verify_and_publish_supporting_profile",
]
