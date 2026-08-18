"""Standalone CRSP numeric/unit discovery and locked-profile compiler.

The module reads only the frozen daily, monthly, and effective-shares CRSP
files.  Row payload is date-gated before Python inspection, retained batches
are bounded, and large joins are performed in an ephemeral SQLite database.
Only aggregate evidence is returned or rendered; raw security keys never leave
the scratch database.

The checked-in empirical profile binds an independently reviewable aggregate
lock. Production compilation authenticates and reproduces that lock before any
quality artifact or pointer may be published. A test-only discovery entry point
remains available only with an explicitly unlocked settings object.
"""

from __future__ import annotations

import csv
import dataclasses
import datetime as dt
import hashlib
import io
import json
import os
import re
import sqlite3
import stat
import tempfile
import tomllib
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN, localcontext
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import pyarrow as pa
import pyarrow.csv as pacsv


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_POLICY_CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "phase_02_canonicalization"
    / "crsp_numeric_unit_policy_v1.toml"
)
DEFAULT_PROFILE_CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "phase_02_canonicalization"
    / "crsp_numeric_unit_profile_development_v1.toml"
)
DEFAULT_EMPIRICAL_LOCK_CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "phase_02_canonicalization"
    / "crsp_numeric_unit_empirical_lock_v1.json"
)
DEFAULT_PATHS_CONFIG = PROJECT_ROOT / "configs" / "paths.toml"
DEFAULT_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "phase_02_canonical_panel"
DEFAULT_RUN_ROOT = PROJECT_ROOT / "runs" / "phase_02_canonical_panel"
LOCKED_POLICY_SHA256 = (
    "9d979aa016bdf6acc90a98b2168e4d1b15cdeb120715168d55a3bb0466d377be"
)
LOCKED_PROFILE_CONFIG_SHA256 = (
    "99469fbd0e2dc6046b45490cad4253942ae11b22b28ed9886e878cdf4e1958c1"
)
BYTE_CHUNK_SIZE = 8 << 20
SMALL_AUTHENTICATED_FILE_MAX_BYTES = 16 << 20
SNAPSHOT_INVENTORY_DOMAIN = b"phase02-crsp-numeric-unit-snapshots-v1\x00"

SOURCE_NAMES = ("daily", "monthly", "shares")
SOURCE_FILENAMES = {
    "daily": "crsp_daily_ciz.csv",
    "monthly": "crsp_monthly_ciz.csv",
    "shares": "crsp_shares_outstanding_ciz.csv",
}
QUALITY_FILE_SET = (
    "_SUCCESS.json",
    "capitalization_diagnostics.csv",
    "checks.csv",
    "downstream_gate_status.json",
    "identity_inventory.csv",
    "input_receipt.json",
    "numeric_inventory.csv",
    "profile_contract.json",
    "return_inventory.csv",
    "summary.json",
)
PRODUCTION_LOCK_KEYS = frozenset(
    {
        "daily_numeric_expectations_locked",
        "daily_return_cross_tabs_locked",
        "daily_price_volume_expectations_locked",
        "monthly_numeric_expectations_locked",
        "monthly_observation_date_expectations_locked",
        "monthly_crosscheck_expectations_locked",
        "shares_join_expectations_locked",
        "capitalization_diagnostics_locked",
        "non_ADR_scale_population_locked",
        "non_ADR_scale_threshold_locked",
        "all_required_locks_true",
    }
)
EMPIRICAL_TABLE_NAMES = (
    "capitalization_diagnostics",
    "identity_inventory",
    "numeric_inventory",
    "return_inventory",
)
MEASUREMENT_EXCLUDED_FIELDS = {
    "numeric_inventory": frozenset({"status", "severity"}),
    "return_inventory": frozenset({"semantic_rule", "status", "severity"}),
    "identity_inventory": frozenset({"severity", "status", "tolerance_rule"}),
    "capitalization_diagnostics": frozenset(
        {"severity", "status", "threshold_rule"}
    ),
}
NON_ADR_COUNT_FIELDS = (
    "population_rows",
    "comparable_rows",
    "excluded_rows",
    "zero_share_match_rows",
    "one_share_match_rows",
    "multiple_share_match_rows",
    "formula_pass_rows",
    "formula_fail_rows",
)


class NumericUnitProfileError(RuntimeError):
    """Raised when a contract, scan, conservation rule, or lock fails."""


@dataclasses.dataclass(frozen=True)
class NumericUnitSettings:
    project_root: Path
    policy_config_path: Path
    profile_config_path: Path
    paths_config_path: Path
    policy: dict[str, Any]
    config: dict[str, Any]
    database_root: Path
    raw_root: Path
    quality_root: Path
    artifact_root: Path
    run_root: Path
    policy_config_sha256: str | None = None
    policy_config_size_bytes: int | None = None
    profile_config_sha256: str | None = None
    profile_config_size_bytes: int | None = None
    paths_config_sha256: str | None = None
    paths_config_size_bytes: int | None = None
    block_size_bytes: int = 1 << 20
    authenticate_inputs: bool = True
    authenticate_upstreams: bool = True
    verified_trading_dates: frozenset[dt.date] | None = None


@dataclasses.dataclass(frozen=True)
class ParsedNumeric:
    raw: str
    value: Decimal
    quantum: Decimal
    lower: Decimal
    upper: Decimal


def canonical_json_bytes(value: Any) -> bytes:
    try:
        rendered = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise NumericUnitProfileError(f"Value is not canonical JSON: {exc}") from exc
    return (rendered + "\n").encode("utf-8")


def _canonical_json_text(value: Any) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise NumericUnitProfileError(f"Value is not canonical JSON: {exc}") from exc


def _absolute(path: str | Path) -> Path:
    return Path(os.path.abspath(os.fspath(Path(path).expanduser())))


def _project_path(path: str | Path) -> Path:
    value = Path(path).expanduser()
    return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()


def _same_operational_file(*values: os.stat_result) -> bool:
    """Return whether every stat describes one unchanged regular file."""
    if not values or any(not stat.S_ISREG(value.st_mode) for value in values):
        return False
    first = values[0]
    return all(
        value.st_dev == first.st_dev
        and value.st_ino == first.st_ino
        and value.st_size == first.st_size
        and value.st_mtime_ns == first.st_mtime_ns
        and value.st_ctime_ns == first.st_ctime_ns
        for value in values[1:]
    )


@dataclasses.dataclass
class _OpenedRegularFile:
    """A no-follow descriptor anchored beneath opened directory descriptors."""

    path: Path
    label: str
    descriptor: int
    before: os.stat_result
    opened: os.stat_result
    closed: bool = False

    def assert_stable(self) -> None:
        if self.closed:
            raise NumericUnitProfileError(
                f"Authenticated {self.label} descriptor is already closed"
            )
        try:
            current = os.fstat(self.descriptor)
            lexical = self.path.lstat()
        except OSError as exc:
            raise NumericUnitProfileError(
                f"Cannot restat authenticated {self.label}"
            ) from exc
        if not _same_operational_file(self.before, self.opened, current, lexical):
            raise NumericUnitProfileError(
                f"Authenticated {self.label} changed while it was in use"
            )

    def close(self, *, validate: bool = True) -> None:
        if self.closed:
            return
        failure: BaseException | None = None
        if validate:
            try:
                self.assert_stable()
            except BaseException as exc:  # preserve validation while closing FDs
                failure = exc
        try:
            os.close(self.descriptor)
        finally:
            self.closed = True
        if failure is not None:
            raise failure


def _open_nofollow_regular_file(path: Path, label: str) -> _OpenedRegularFile:
    """Open a regular file with no-follow flags and explicit component checks."""
    path = _absolute(path)
    if not path.is_absolute() or not path.name:
        raise NumericUnitProfileError(f"Authenticated {label} path is invalid")
    file_flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current = current / component
        try:
            metadata = current.lstat()
        except OSError as exc:
            raise NumericUnitProfileError(
                f"Cannot stat authenticated {label} path component"
            ) from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise NumericUnitProfileError(
                f"Symlink is prohibited in authenticated {label} path"
            )
    try:
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode):
            raise NumericUnitProfileError(
                f"Authenticated {label} is not a regular file"
            )
        try:
            descriptor = os.open(path, file_flags)
        except OSError as exc:
            raise NumericUnitProfileError(
                f"Cannot open authenticated {label} without following symlinks"
            ) from exc
        try:
            opened = os.fstat(descriptor)
            if not _same_operational_file(before, opened):
                raise NumericUnitProfileError(
                    f"Authenticated {label} changed while it was opened"
                )
        except BaseException:
            os.close(descriptor)
            raise
        return _OpenedRegularFile(
            path=path,
            label=label,
            descriptor=descriptor,
            before=before,
            opened=opened,
        )
    except BaseException:
        raise


def _hash_descriptor(descriptor: int, chunk_size: int = BYTE_CHUNK_SIZE) -> str:
    digest = hashlib.sha256()
    os.lseek(descriptor, 0, os.SEEK_SET)
    while chunk := os.read(descriptor, chunk_size):
        digest.update(chunk)
    return digest.hexdigest()


def _hash_authenticated_regular_file(
    path: Path, label: str, chunk_size: int = BYTE_CHUNK_SIZE
) -> tuple[str, int]:
    opened = _open_nofollow_regular_file(path, label)
    try:
        digest = _hash_descriptor(opened.descriptor, chunk_size)
        opened.assert_stable()
        return digest, int(opened.opened.st_size)
    finally:
        opened.close(validate=False)


def _read_authenticated_regular_file(
    path: Path,
    label: str,
    *,
    max_size_bytes: int = SMALL_AUTHENTICATED_FILE_MAX_BYTES,
) -> tuple[bytes, str, int]:
    opened = _open_nofollow_regular_file(path, label)
    try:
        size = int(opened.opened.st_size)
        if size > max_size_bytes:
            raise NumericUnitProfileError(
                f"Authenticated {label} exceeds the bounded in-memory read limit"
            )
        digest = hashlib.sha256()
        chunks: list[bytes] = []
        os.lseek(opened.descriptor, 0, os.SEEK_SET)
        while chunk := os.read(opened.descriptor, 1 << 20):
            digest.update(chunk)
            chunks.append(chunk)
        opened.assert_stable()
        return b"".join(chunks), digest.hexdigest(), size
    finally:
        opened.close(validate=False)


def sha256_file(path: Path, chunk_size: int = BYTE_CHUNK_SIZE) -> str:
    digest, _ = _hash_authenticated_regular_file(
        path, f"file {Path(path).name}", chunk_size
    )
    return digest


def _json_object_from_bytes(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise NumericUnitProfileError(f"Cannot read JSON object {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise NumericUnitProfileError(f"Expected JSON object: {label}")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    payload, _, _ = _read_authenticated_regular_file(path, f"JSON {path.name}")
    return _json_object_from_bytes(payload, str(path))


def _safe_project_relative_path(
    settings: NumericUnitSettings, value: str | Path, label: str
) -> tuple[Path, str]:
    text = os.fspath(value)
    relative = Path(text)
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
        or "\\" in text
    ):
        raise NumericUnitProfileError(f"Unsafe authenticated {label} path")
    project_root = _absolute(settings.project_root)
    path = _absolute(project_root / relative)
    try:
        path.relative_to(project_root)
    except ValueError as exc:
        raise NumericUnitProfileError(
            f"Authenticated {label} path escapes project_root"
        ) from exc
    return path, relative.as_posix()


def _bound_settings_file_receipts(
    settings: NumericUnitSettings,
) -> dict[str, tuple[str, int]]:
    """Authenticate the current files against the exact bytes parsed at load."""
    specifications = (
        (
            "policy",
            settings.policy_config_path,
            settings.policy_config_sha256,
            settings.policy_config_size_bytes,
        ),
        (
            "profile",
            settings.profile_config_path,
            settings.profile_config_sha256,
            settings.profile_config_size_bytes,
        ),
        (
            "paths",
            settings.paths_config_path,
            settings.paths_config_sha256,
            settings.paths_config_size_bytes,
        ),
    )
    receipts: dict[str, tuple[str, int]] = {}
    for label, path, expected_sha, expected_size in specifications:
        digest, size = _hash_authenticated_regular_file(
            path, f"numeric-unit {label} config"
        )
        if expected_sha is not None and digest != expected_sha:
            raise NumericUnitProfileError(
                f"Numeric-unit {label} config changed after it was parsed"
            )
        if expected_size is not None and size != expected_size:
            raise NumericUnitProfileError(
                f"Numeric-unit {label} config size changed after it was parsed"
            )
        receipts[label] = (digest, size)
    return receipts


def load_settings(
    policy_config_path: Path = DEFAULT_POLICY_CONFIG,
    profile_config_path: Path = DEFAULT_PROFILE_CONFIG,
    paths_config_path: Path = DEFAULT_PATHS_CONFIG,
    *,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
    run_root: Path = DEFAULT_RUN_ROOT,
    block_size_bytes: int = 1 << 20,
    authenticate_inputs: bool = True,
    authenticate_upstreams: bool = True,
) -> NumericUnitSettings:
    """Load and bind the current policy/profile without scanning raw data."""
    policy_config_path = _absolute(policy_config_path)
    profile_config_path = _absolute(profile_config_path)
    paths_config_path = _absolute(paths_config_path)
    try:
        policy_payload, policy_sha, policy_size = _read_authenticated_regular_file(
            policy_config_path, "numeric-unit policy"
        )
        profile_payload, profile_sha, profile_size = _read_authenticated_regular_file(
            profile_config_path, "numeric-unit profile"
        )
        paths_payload, paths_sha, paths_size = _read_authenticated_regular_file(
            paths_config_path, "paths config"
        )
        policy = tomllib.loads(policy_payload.decode("utf-8"))
        config = tomllib.loads(profile_payload.decode("utf-8"))
        paths = tomllib.loads(paths_payload.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError, KeyError) as exc:
        raise NumericUnitProfileError(f"Cannot load numeric-unit settings: {exc}") from exc
    if policy_sha != LOCKED_POLICY_SHA256:
        raise NumericUnitProfileError(
            f"Numeric-unit policy SHA differs: expected {LOCKED_POLICY_SHA256}, "
            f"actual {policy_sha}"
        )
    if config.get("policy", {}).get("sha256") != policy_sha:
        raise NumericUnitProfileError("Empirical profile does not bind the current policy SHA")
    if (
        profile_config_path == DEFAULT_PROFILE_CONFIG.resolve()
        and profile_sha != LOCKED_PROFILE_CONFIG_SHA256
    ):
        raise NumericUnitProfileError(
            "Checked-in numeric-unit profile SHA differs: expected "
            f"{LOCKED_PROFILE_CONFIG_SHA256}, actual {profile_sha}"
        )
    if policy.get("status") != config["policy"]["required_status"]:
        raise NumericUnitProfileError("Policy status differs from empirical-profile requirement")
    actual_files = tuple(sorted(config["outputs"]["closed_quality_file_set"]))
    if actual_files != tuple(sorted(QUALITY_FILE_SET)):
        raise NumericUnitProfileError("Numeric-unit closed quality file set differs")
    if set(config.get("sources", {})) != set(SOURCE_NAMES):
        raise NumericUnitProfileError(
            "Numeric-unit config must name exactly daily, monthly, and shares"
        )
    for source, filename in SOURCE_FILENAMES.items():
        if config["sources"][source].get("filename") != filename:
            raise NumericUnitProfileError(
                f"Numeric-unit {source} filename differs from the frozen direct basename"
            )
    if config.get("status") == "discovery_only_unlocked":
        prohibited = {
            key: value
            for key, value in config["outputs"].items()
            if key.startswith("discovery_may_write_") and value is not False
        }
        if prohibited:
            raise NumericUnitProfileError(
                f"Unlocked discovery output prohibitions are not fail-closed: {prohibited}"
            )
        if config["downstream"][
            "current_unlocked_config_authorizes_any_downstream_action"
        ] is not False:
            raise NumericUnitProfileError("Unlocked discovery cannot authorize downstream use")
    configured = Path(paths["database"]["root"]).expanduser()
    database_root = _absolute(os.environ.get("PORTFOLIO_DATABASE_ROOT", configured))
    namespace = "phase_02_canonical_panel"
    return NumericUnitSettings(
        project_root=PROJECT_ROOT,
        policy_config_path=policy_config_path,
        profile_config_path=profile_config_path,
        paths_config_path=paths_config_path,
        policy=policy,
        config=config,
        database_root=database_root,
        raw_root=_absolute(database_root / paths["database"]["raw"]),
        quality_root=_absolute(
            database_root / paths["database"]["quality"] / namespace
        ),
        artifact_root=_absolute(artifact_root),
        run_root=_absolute(run_root),
        policy_config_sha256=policy_sha,
        policy_config_size_bytes=policy_size,
        profile_config_sha256=profile_sha,
        profile_config_size_bytes=profile_size,
        paths_config_sha256=paths_sha,
        paths_config_size_bytes=paths_size,
        block_size_bytes=max(64 << 10, int(block_size_bytes)),
        authenticate_inputs=authenticate_inputs,
        authenticate_upstreams=authenticate_upstreams,
    )


def _fullmatch(pattern: str, value: str) -> bool:
    return re.fullmatch(f"(?:{pattern})", value, flags=re.ASCII) is not None


def decimal_quantum(value: str) -> Decimal:
    """Return the displayed last-digit quantum for a valid decimal lexeme."""
    try:
        parsed = Decimal(value)
    except InvalidOperation as exc:
        raise NumericUnitProfileError(f"Invalid Decimal lexeme {value!r}") from exc
    if not parsed.is_finite():
        raise NumericUnitProfileError(f"Nonfinite Decimal lexeme {value!r}")
    return Decimal((0, (1,), parsed.as_tuple().exponent))


def _operation_precision(*values: Decimal, extra_digits: int = 10) -> int:
    if not values:
        return 50
    adjusted = [value.adjusted() if value else 0 for value in values]
    digit_total = sum(len(value.as_tuple().digits) for value in values)
    exponent_span = max(adjusted) - min(adjusted)
    return max(50, digit_total + exponent_span + extra_digits)


def _exact_multiply(left: Decimal, right: Decimal) -> Decimal:
    with localcontext() as context:
        context.prec = _operation_precision(left, right)
        return left * right


def _exact_add(left: Decimal, right: Decimal) -> Decimal:
    with localcontext() as context:
        context.prec = _operation_precision(left, right)
        return left + right


def _exact_subtract(left: Decimal, right: Decimal) -> Decimal:
    with localcontext() as context:
        context.prec = _operation_precision(left, right)
        return left - right


def _diagnostic_divide(numerator: Decimal, denominator: Decimal) -> Decimal:
    if denominator == 0:
        raise NumericUnitProfileError("Diagnostic Decimal division by zero")
    with localcontext() as context:
        context.prec = max(
            50,
            len(numerator.as_tuple().digits)
            + len(denominator.as_tuple().digits)
            + 20,
        )
        return numerator / denominator


def _relative_error(absolute: Decimal, *reference_values: Decimal) -> Decimal | None:
    denominator = max((abs(value) for value in reference_values), default=Decimal(0))
    return _diagnostic_divide(absolute, denominator) if denominator else None


def parse_numeric_lexeme(
    value: str,
    *,
    grammar: str,
    interval_kind: str = "reported",
) -> ParsedNumeric | None:
    """Strictly parse a decoded field; blank is the only numeric null."""
    if value == "":
        return None
    if not _fullmatch(grammar, value):
        raise NumericUnitProfileError(f"Malformed numeric lexeme {value!r}")
    try:
        parsed = Decimal(value)
    except InvalidOperation as exc:
        raise NumericUnitProfileError(f"Malformed numeric lexeme {value!r}") from exc
    if not parsed.is_finite():
        raise NumericUnitProfileError(f"Nonfinite numeric lexeme {value!r}")
    if interval_kind == "point":
        quantum = Decimal(0)
        lower = upper = parsed
    elif interval_kind == "shares_thousand":
        quantum = Decimal(1)
        lower = _exact_subtract(parsed, Decimal("0.5"))
        upper = _exact_add(parsed, Decimal("0.5"))
    elif interval_kind == "reported":
        quantum = decimal_quantum(value)
        with localcontext() as context:
            context.prec = _operation_precision(parsed, quantum)
            half = quantum / 2
        lower = _exact_subtract(parsed, half)
        upper = _exact_add(parsed, half)
    else:
        raise NumericUnitProfileError(f"Unknown interval kind {interval_kind!r}")
    return ParsedNumeric(value, parsed, quantum, lower, upper)


def multiply_intervals(
    left: tuple[Decimal, Decimal], right: tuple[Decimal, Decimal]
) -> tuple[Decimal, Decimal]:
    products = (
        _exact_multiply(left[0], right[0]),
        _exact_multiply(left[0], right[1]),
        _exact_multiply(left[1], right[0]),
        _exact_multiply(left[1], right[1]),
    )
    return min(products), max(products)


def intervals_overlap(
    left: tuple[Decimal, Decimal], right: tuple[Decimal, Decimal]
) -> bool:
    return left[0] <= right[1] and right[0] <= left[1]


def quantized_product_matches(
    left: ParsedNumeric, right: ParsedNumeric, reported: ParsedNumeric
) -> bool:
    calculated = multiply_intervals(
        (left.lower, left.upper), (right.lower, right.upper)
    )
    return intervals_overlap(calculated, (reported.lower, reported.upper))


def _wire(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, (dict, list, tuple)):
        return _canonical_json_text(value)
    return str(value)


def _csv_bytes(fields: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> bytes:
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(
        handle,
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
    return handle.getvalue().encode("utf-8")


def _strict_date(value: str, label: str) -> dt.date:
    if len(value) != 10 or value[4:5] != "-" or value[7:8] != "-":
        raise NumericUnitProfileError(f"{label} is not strict YYYY-MM-DD")
    try:
        parsed = dt.date.fromisoformat(value)
    except ValueError as exc:
        raise NumericUnitProfileError(f"{label} is not a Gregorian date") from exc
    if parsed.isoformat() != value:
        raise NumericUnitProfileError(f"{label} is not canonical YYYY-MM-DD")
    return parsed


def _positive_int(value: str, label: str) -> int:
    if not _fullmatch(r"[1-9][0-9]*", value):
        raise NumericUnitProfileError(f"{label} is not a positive ASCII integer")
    maximum = "9223372036854775807"
    if len(value) > len(maximum) or (len(value) == len(maximum) and value > maximum):
        raise NumericUnitProfileError(f"{label} exceeds signed int64")
    parsed = int(value)
    return parsed


def _decimal_text(value: Decimal | None, *, fractional_digits: int | None = None) -> str:
    """Render a finite Decimal without exponent notation or negative zero."""
    if value is None:
        return ""
    if not value.is_finite():
        raise NumericUnitProfileError("Nonfinite Decimal cannot be rendered")
    if value.is_zero():
        return "0"
    if fractional_digits is not None:
        quantum = Decimal(1).scaleb(-fractional_digits)
        with localcontext() as context:
            context.prec = max(50, len(value.as_tuple().digits) + abs(value.adjusted()) + 20)
            value = value.quantize(quantum, rounding=ROUND_HALF_EVEN)
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return "0" if rendered in {"-0", "+0", ""} else rendered


def _rate(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return ""
    with localcontext() as context:
        context.prec = 50
        return _decimal_text(
            Decimal(numerator) / Decimal(denominator), fractional_digits=12
        )


def _segment(value: dt.date, config: Mapping[str, Any]) -> str:
    if value.year == 2014 and value.month == 12:
        return str(config["scope"]["december_period_segment"])
    return str(config["scope"]["development_period_segment"])


def _json_dimension(value: Mapping[str, Any] | None = None) -> str:
    return _canonical_json_text(dict(value or {}))


@dataclasses.dataclass(frozen=True)
class ClassifiedNumeric:
    state: str
    parsed: ParsedNumeric | None
    negative_zero: bool = False


def _classify_numeric(
    raw: str,
    *,
    grammar: str,
    numeric_kind: str,
    domain: str,
) -> ClassifiedNumeric:
    """Classify one lexeme without silently coercing any invalid value."""
    if raw == "":
        return ClassifiedNumeric("blank", None)
    if not _fullmatch(grammar, raw):
        lowered = raw.lower()
        state = "nonfinite" if lowered in {
            "nan", "+nan", "-nan", "inf", "+inf", "-inf", "infinity",
            "+infinity", "-infinity",
        } else "malformed"
        return ClassifiedNumeric(state, None)
    try:
        value = Decimal(raw)
    except InvalidOperation:
        return ClassifiedNumeric("malformed", None)
    if not value.is_finite():
        return ClassifiedNumeric("nonfinite", None)
    if numeric_kind != "decimal" and value > Decimal(9_223_372_036_854_775_807):
        return ClassifiedNumeric("overflow", None)
    interval_kind = (
        "shares_thousand"
        if numeric_kind == "positive_integer_when_present"
        else "point" if numeric_kind == "nonnegative_integer" else "reported"
    )
    parsed = parse_numeric_lexeme(raw, grammar=grammar, interval_kind=interval_kind)
    assert parsed is not None
    in_domain = True
    if domain in {"price", "quote", "capitalization", "shares"}:
        in_domain = value > 0
    elif domain in {"nonnegative", "price_volume"}:
        in_domain = value >= 0
    elif domain == "return":
        in_domain = value >= -1
    return ClassifiedNumeric(
        "valid_in_domain" if in_domain else "valid_out_of_domain",
        parsed,
        value.is_zero() and value.is_signed(),
    )


@dataclasses.dataclass
class _NumericCell:
    population: int = 0
    states: Counter[str] = dataclasses.field(default_factory=Counter)
    negative_zero: int = 0
    minimum: Decimal | None = None
    maximum: Decimal | None = None
    quanta: Counter[str] = dataclasses.field(default_factory=Counter)

    def add(self, classified: ClassifiedNumeric) -> None:
        self.population += 1
        self.states[classified.state] += 1
        if classified.negative_zero:
            self.negative_zero += 1
        if classified.parsed is not None:
            value = classified.parsed.value
            self.minimum = value if self.minimum is None else min(self.minimum, value)
            self.maximum = value if self.maximum is None else max(self.maximum, value)
            self.quanta[_decimal_text(classified.parsed.quantum)] += 1


class _NumericInventory:
    def __init__(self, policy: Mapping[str, Any]) -> None:
        self.policy = policy
        self.cells: dict[tuple[Any, ...], _NumericCell] = defaultdict(_NumericCell)

    def classify(self, source: str, column: str, raw: str) -> ClassifiedNumeric:
        parsing = self.policy["parsing"]
        daily = self.policy["fields"]["daily"]
        monthly = self.policy["fields"]["monthly"]
        if source == "shares" or column in monthly.get(
            "positive_integer_when_present_columns", []
        ):
            kind = "positive_integer_when_present"
            grammar = parsing["positive_integer_grammar"]
            domain = "shares"
        elif column in daily.get("nonnegative_integer_columns", []) or column in monthly.get(
            "nonnegative_integer_columns", []
        ):
            kind = "nonnegative_integer"
            grammar = parsing["nonnegative_integer_grammar"]
            domain = "nonnegative"
        else:
            kind = "decimal"
            grammar = parsing["decimal_grammar"]
            if column in daily.get("return_columns", []) or column in monthly.get(
                "return_columns", []
            ):
                domain = "return"
            elif column in daily.get("price_and_quote_columns", []) or column == "MthPrc":
                domain = "price"
            elif column in daily.get("capitalization_columns", []) or column == "MthCap":
                domain = "capitalization"
            elif column in {"DlyPrcVol", "MthPrcVol"}:
                domain = "price_volume"
            else:
                domain = "unbounded"
        return _classify_numeric(
            raw, grammar=grammar, numeric_kind=kind, domain=domain
        )

    def observe(
        self,
        *,
        source: str,
        column: str,
        date: dt.date,
        stratum: Mapping[str, Any],
        raw: str,
    ) -> ClassifiedNumeric:
        classified = self.classify(source, column, raw)
        field_config = self.policy["fields"]["daily" if source == "daily" else "monthly"] if source != "shares" else None
        if source == "shares" or (
            field_config is not None
            and column in field_config.get("positive_integer_when_present_columns", [])
        ):
            kind = "positive_integer_when_present"
        elif field_config is not None and column in field_config.get(
            "nonnegative_integer_columns", []
        ):
            kind = "nonnegative_integer"
        else:
            kind = "decimal"
        key = (
            source,
            column,
            date.year,
            _segment(date, {"scope": {
                "december_period_segment": "embargo_2014_12",
                "development_period_segment": "development_2000_2014_11",
            }}),
            _json_dimension(stratum),
            kind,
        )
        self.cells[key].add(classified)
        return classified

    def rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        hard_outside = {
            "DlyRet", "DlyRetx", "DlyVol", "MthRet", "MthRetx", "MthDisCnt",
            "MthVol", "MthPrcVolMissCnt", "ShrOut",
        }
        for key, cell in self.cells.items():
            source, column, year, segment, stratum, kind = key
            hard_failures = (
                cell.states["malformed"]
                + cell.states["nonfinite"]
                + cell.states["overflow"]
                + (cell.states["valid_out_of_domain"] if column in hard_outside else 0)
            )
            hard_rule = column in hard_outside
            diagnostic = cell.states["valid_out_of_domain"] and not hard_failures
            rows.append(
                {
                    "source": source,
                    "column": column,
                    "year": year,
                    "period_segment": segment,
                    "stratum": stratum,
                    "numeric_kind": kind,
                    "population_rows": cell.population,
                    "blank_rows": cell.states["blank"],
                    "valid_in_domain_rows": cell.states["valid_in_domain"],
                    "valid_out_of_domain_rows": cell.states["valid_out_of_domain"],
                    "malformed_rows": cell.states["malformed"],
                    "nonfinite_rows": cell.states["nonfinite"],
                    "overflow_rows": cell.states["overflow"],
                    "negative_zero_rows": cell.negative_zero,
                    "min_value": _decimal_text(cell.minimum),
                    "max_value": _decimal_text(cell.maximum),
                    "quantum_counts": _canonical_json_text(dict(sorted(cell.quanta.items()))),
                    "status": "FAIL" if hard_failures else "DIAGNOSTIC" if diagnostic else "PASS",
                    "severity": "hard" if hard_rule or hard_failures else "diagnostic",
                }
            )
        return rows


@dataclasses.dataclass
class _IdentityCell:
    population: int = 0
    comparable: int = 0
    passed: int = 0
    failed: int = 0
    exclusions: Counter[str] = dataclasses.field(default_factory=Counter)
    max_absolute_error: Decimal | None = None
    max_relative_error: Decimal | None = None


class _IdentityInventory:
    def __init__(self) -> None:
        self.cells: dict[tuple[Any, ...], _IdentityCell] = defaultdict(_IdentityCell)

    def exclude(self, key: tuple[Any, ...], reason: str) -> None:
        cell = self.cells[key]
        cell.population += 1
        cell.exclusions[reason] += 1

    def compare(
        self,
        key: tuple[Any, ...],
        passed: bool,
        *,
        absolute_error: Decimal | None = None,
        relative_error: Decimal | None = None,
    ) -> None:
        cell = self.cells[key]
        cell.population += 1
        cell.comparable += 1
        if passed:
            cell.passed += 1
        else:
            cell.failed += 1
        if absolute_error is not None:
            cell.max_absolute_error = (
                absolute_error if cell.max_absolute_error is None
                else max(cell.max_absolute_error, absolute_error)
            )
        if relative_error is not None:
            cell.max_relative_error = (
                relative_error if cell.max_relative_error is None
                else max(cell.max_relative_error, relative_error)
            )

    def rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for key, cell in self.cells.items():
            identity_id, source, year, segment, stratum, severity, tolerance = key
            excluded = sum(cell.exclusions.values())
            if cell.population != cell.comparable + excluded:
                raise NumericUnitProfileError("Identity population conservation failed")
            if cell.comparable != cell.passed + cell.failed:
                raise NumericUnitProfileError("Identity comparable conservation failed")
            status = (
                "FAIL"
                if severity == "hard" and cell.failed
                else "DIAGNOSTIC"
                if cell.failed or (cell.population > 0 and cell.comparable == 0)
                else "PASS"
            )
            rows.append(
                {
                    "identity_id": identity_id,
                    "source": source,
                    "year": year,
                    "period_segment": segment,
                    "stratum": stratum,
                    "severity": severity,
                    "population_rows": cell.population,
                    "comparable_rows": cell.comparable,
                    "pass_rows": cell.passed,
                    "fail_rows": cell.failed,
                    "excluded_rows": excluded,
                    "exclusions": _canonical_json_text(dict(sorted(cell.exclusions.items()))),
                    "coverage_rate": _rate(cell.comparable, cell.population),
                    "pass_rate": _rate(cell.passed, cell.comparable),
                    "max_absolute_error": _decimal_text(cell.max_absolute_error),
                    "max_relative_error": _decimal_text(cell.max_relative_error),
                    "tolerance_rule": tolerance,
                    "status": status,
                }
            )
        return rows


def _identity_zero_comparable_active_segments(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return hard-identity active segments with no comparisons across strata."""
    pooled: dict[tuple[str, str, int, str], list[int]] = defaultdict(
        lambda: [0, 0]
    )
    for row in rows:
        if row["severity"] != "hard":
            continue
        key = (
            str(row["identity_id"]),
            str(row["source"]),
            int(row["year"]),
            str(row["period_segment"]),
        )
        pooled[key][0] += int(row["population_rows"])
        pooled[key][1] += int(row["comparable_rows"])
    failures = [
        {
            "identity_id": identity_id,
            "source": source,
            "year": year,
            "period_segment": segment,
            "population_rows": population,
            "comparable_rows": comparable,
        }
        for (identity_id, source, year, segment), (population, comparable)
        in pooled.items()
        if population > 0 and comparable == 0
    ]
    return sorted(
        failures,
        key=lambda row: (
            str(row["identity_id"]).encode("utf-8"),
            str(row["source"]).encode("utf-8"),
            int(row["year"]),
            str(row["period_segment"]).encode("utf-8"),
        ),
    )


_IDENTITY_REFERENCE_COLUMNS = {
    "daily_price_volume": ("daily", "daily", "DlyPrc"),
    "monthly_daily_price": ("monthly_daily", "monthly", "MthPrc"),
    "monthly_daily_cap": ("monthly_daily", "monthly", "MthCap"),
}


def _identity_active_segment_population_differences(
    numeric_rows: Sequence[Mapping[str, Any]],
    identity_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Compare hard-identity audit populations with canonical numeric grains."""
    expected: Counter[tuple[str, str, int, str]] = Counter()
    reference_columns = {
        (numeric_source, reference_column)
        for _identity_source, numeric_source, reference_column
        in _IDENTITY_REFERENCE_COLUMNS.values()
    }
    for row in numeric_rows:
        source = str(row["source"])
        column = str(row["column"])
        if (source, column) not in reference_columns:
            continue
        expected[
            (source, column, int(row["year"]), str(row["period_segment"]))
        ] += int(row["population_rows"])

    actual: Counter[tuple[str, str, int, str]] = Counter()
    for row in identity_rows:
        identity_id = str(row["identity_id"])
        if identity_id not in _IDENTITY_REFERENCE_COLUMNS:
            continue
        actual[
            (
                identity_id,
                str(row["source"]),
                int(row["year"]),
                str(row["period_segment"]),
            )
        ] += int(row["population_rows"])

    differences: list[dict[str, Any]] = []
    for identity_id, (
        identity_source,
        numeric_source,
        column,
    ) in _IDENTITY_REFERENCE_COLUMNS.items():
        groups = {
            (year, segment)
            for expected_source, expected_column, year, segment in expected
            if expected_source == numeric_source and expected_column == column
        } | {
            (year, segment)
            for actual_identity, actual_source, year, segment in actual
            if actual_identity == identity_id and actual_source == identity_source
        }
        for year, segment in groups:
            expected_rows = expected[(numeric_source, column, year, segment)]
            actual_rows = actual[(identity_id, identity_source, year, segment)]
            if expected_rows == actual_rows:
                continue
            differences.append(
                {
                    "identity_id": identity_id,
                    "source": identity_source,
                    "year": year,
                    "period_segment": segment,
                    "expected_population_rows": expected_rows,
                    "actual_population_rows": actual_rows,
                }
            )
    return sorted(
        differences,
        key=lambda row: (
            str(row["identity_id"]).encode("utf-8"),
            str(row["source"]).encode("utf-8"),
            int(row["year"]),
            str(row["period_segment"]).encode("utf-8"),
        ),
    )


@dataclasses.dataclass
class _ReturnCell:
    population: int = 0
    coherent: int | None = None
    incoherent: int | None = None


class _ReturnInventory:
    def __init__(self, policy: Mapping[str, Any]) -> None:
        self.policy = policy
        self.cells: dict[tuple[Any, ...], _ReturnCell] = defaultdict(_ReturnCell)

    @staticmethod
    def _state(classified: ClassifiedNumeric) -> str:
        return classified.state

    def observe_daily(
        self,
        *,
        date: dt.date,
        flag: str,
        duration: str,
        total: ClassifiedNumeric,
        returnx: ClassifiedNumeric,
    ) -> None:
        durations = self.policy["return_duration"]
        mapped = durations.get(duration, durations["unknown_period_sentinel"])
        locked_rule = flag == self.policy["return_availability"]["daily"][
            "available_missing_flag"
        ]
        rule = (
            "NA_requires_valid_DlyRet_and_duration_not_MR"
            if locked_rule
            else "discovery_inventory_only_flag_specific_mapping_unresolved"
        )
        key = (
            "daily",
            "daily",
            date.year,
            "embargo_2014_12" if date.year == 2014 and date.month == 12 else "development_2000_2014_11",
            flag,
            duration,
            self._state(total),
            self._state(returnx),
            mapped,
            rule,
        )
        cell = self.cells[key]
        cell.population += 1
        if locked_rule:
            if cell.coherent is None:
                cell.coherent = cell.incoherent = 0
            coherent = total.state == "valid_in_domain" and duration != "MR"
            if coherent:
                cell.coherent += 1
            else:
                cell.incoherent += 1

    def observe_monthly(
        self,
        *,
        date: dt.date,
        flag: str,
        total: ClassifiedNumeric,
        returnx: ClassifiedNumeric,
    ) -> None:
        key = (
            "monthly",
            "monthly",
            date.year,
            "embargo_2014_12" if date.year == 2014 and date.month == 12 else "development_2000_2014_11",
            flag,
            "",
            self._state(total),
            self._state(returnx),
            "",
            "preserve_observed_values_including_numeric_GP_inventory_only",
        )
        self.cells[key].population += 1

    def rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for key, cell in self.cells.items():
            (
                source,
                periodicity,
                year,
                segment,
                flag,
                duration,
                total_state,
                returnx_state,
                mapped,
                rule,
            ) = key
            locked = cell.coherent is not None
            incoherent = cell.incoherent if locked else ""
            coherent = cell.coherent if locked else ""
            rows.append(
                {
                    "source": source,
                    "periodicity": periodicity,
                    "year": year,
                    "period_segment": segment,
                    "return_flag": flag,
                    "duration_flag": duration,
                    "total_return_state": total_state,
                    "returnx_state": returnx_state,
                    "mapped_trading_periods": mapped,
                    "semantic_rule": rule,
                    "population_rows": cell.population,
                    "coherent_rows": coherent,
                    "incoherent_rows": incoherent,
                    "status": (
                        "FAIL" if locked and incoherent
                        else "PASS" if locked
                        else "DIAGNOSTIC"
                    ),
                    "severity": "hard" if locked else "diagnostic",
                }
            )
        return rows


@dataclasses.dataclass
class _CapCell:
    population: int = 0
    comparable: int = 0
    passed: int = 0
    failed: int = 0
    zero_matches: int = 0
    one_matches: int = 0
    multiple_matches: int = 0
    exclusions: Counter[str] = dataclasses.field(default_factory=Counter)
    ratio_count: int = 0


class _CapitalizationInventory:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection
        self.cells: dict[tuple[Any, ...], _CapCell] = defaultdict(_CapCell)
        connection.execute(
            "CREATE TABLE cap_security_lines (group_key TEXT NOT NULL, permno INTEGER NOT NULL, PRIMARY KEY(group_key, permno)) WITHOUT ROWID"
        )
        connection.execute(
            "CREATE TABLE cap_ratios (group_key TEXT NOT NULL, adjusted_exponent INTEGER NOT NULL, significand_digits TEXT NOT NULL, ratio TEXT NOT NULL)"
        )

    def observe(
        self,
        *,
        formula_id: str | None = None,
        periodicity: str,
        date: dt.date,
        primary_exchange: str,
        share_type: str,
        adr_flag: str,
        permno: int,
        row_token: str,
        match_count: int,
        cap: ClassifiedNumeric,
        price: ClassifiedNumeric,
        shares: ClassifiedNumeric | None,
    ) -> None:
        legacy_formula = {
            "daily": ("daily_interval_cap_price_shares", "daily"),
            "monthly_embedded": ("monthly_embedded_cap_price_shares", "monthly"),
            "monthly_interval": ("monthly_interval_cap_price_shares", "monthly"),
            "monthly": ("monthly_interval_cap_price_shares", "monthly"),
        }
        if formula_id is None:
            formula_id, periodicity = legacy_formula[periodicity]
        key = (
            formula_id,
            periodicity,
            date.year,
            "embargo_2014_12" if date.year == 2014 and date.month == 12 else "development_2000_2014_11",
            primary_exchange,
            share_type,
            adr_flag,
        )
        cell = self.cells[key]
        cell.population += 1
        group_key = _canonical_json_text(list(key))
        self.connection.execute(
            "INSERT OR IGNORE INTO cap_security_lines(group_key, permno) VALUES (?, ?)",
            (group_key, permno),
        )
        if match_count == 0:
            cell.zero_matches += 1
        elif match_count == 1:
            cell.one_matches += 1
        else:
            cell.multiple_matches += 1

        reason = ""
        if match_count > 1:
            reason = "multiple_share_interval_matches"
        elif formula_id != "monthly_embedded_cap_price_shares" and match_count == 0:
            reason = "zero_share_interval_matches"
        elif cap.state != "valid_in_domain":
            reason = "vendor_cap_missing_or_invalid"
        elif price.state != "valid_in_domain":
            reason = "price_missing_or_invalid"
        elif shares is None or shares.state != "valid_in_domain":
            reason = "shares_missing_or_invalid"
        if reason:
            cell.exclusions[reason] += 1
            return

        assert cap.parsed is not None and price.parsed is not None
        assert shares is not None and shares.parsed is not None
        cell.comparable += 1
        passed = quantized_product_matches(price.parsed, shares.parsed, cap.parsed)
        if passed:
            cell.passed += 1
        else:
            cell.failed += 1
        denominator = _exact_multiply(price.parsed.value, shares.parsed.value)
        if denominator != 0:
            with localcontext() as context:
                context.prec = max(
                    50,
                    len(cap.parsed.value.as_tuple().digits)
                    + len(denominator.as_tuple().digits)
                    + 20,
                )
                ratio = cap.parsed.value / denominator
            ratio_tuple = ratio.as_tuple()
            ratio_digits = list(ratio_tuple.digits)
            while len(ratio_digits) > 1 and ratio_digits[-1] == 0:
                ratio_digits.pop()
            digits = "".join(str(digit) for digit in ratio_digits)
            self.connection.execute(
                "INSERT INTO cap_ratios VALUES (?, ?, ?, ?)",
                (group_key, ratio.adjusted(), digits, _decimal_text(ratio)),
            )
            cell.ratio_count += 1

    def _quantile(self, group_key: str, fraction: Decimal) -> str:
        count = self.connection.execute(
            "SELECT COUNT(*) FROM cap_ratios WHERE group_key = ?", (group_key,)
        ).fetchone()[0]
        if count == 0:
            return ""
        with localcontext() as context:
            context.prec = 50
            position = fraction * Decimal(count - 1)
        lower_index = int(position)
        upper_index = min(lower_index + 1, count - 1)
        values: list[Decimal] = []
        for index in (lower_index, upper_index):
            row = self.connection.execute(
                """
                SELECT ratio FROM cap_ratios
                WHERE group_key = ?
                ORDER BY adjusted_exponent, significand_digits, ratio
                LIMIT 1 OFFSET ?
                """,
                (group_key, index),
            ).fetchone()
            if row is None:
                raise NumericUnitProfileError("Exact capitalization quantile lookup failed")
            values.append(Decimal(row[0]))
        weight = position - Decimal(lower_index)
        interpolated = _exact_add(
            values[0], _exact_multiply(_exact_subtract(values[1], values[0]), weight)
        )
        return _decimal_text(interpolated, fractional_digits=12)

    def rows(self) -> list[dict[str, Any]]:
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS cap_ratios_order ON cap_ratios(group_key, adjusted_exponent, significand_digits, ratio)"
        )
        line_counts = dict(
            self.connection.execute(
                "SELECT group_key, COUNT(*) FROM cap_security_lines GROUP BY group_key"
            )
        )
        rows: list[dict[str, Any]] = []
        for key, cell in self.cells.items():
            excluded = sum(cell.exclusions.values())
            if cell.population != cell.comparable + excluded:
                raise NumericUnitProfileError("Capitalization population conservation failed")
            if cell.comparable != cell.passed + cell.failed:
                raise NumericUnitProfileError("Capitalization comparable conservation failed")
            formula_id, periodicity, year, segment, exchange, share_type, adr = key
            hard = cell.multiple_matches > 0
            status = "FAIL" if hard else "DIAGNOSTIC"
            group_key = _canonical_json_text(list(key))
            rows.append(
                {
                    "formula_id": formula_id,
                    "periodicity": periodicity,
                    "year": year,
                    "period_segment": segment,
                    "primary_exchange": exchange,
                    "share_type": share_type,
                    "shr_adr_flag": adr,
                    "security_line_count": line_counts.get(group_key, 0),
                    "population_rows": cell.population,
                    "comparable_rows": cell.comparable,
                    "excluded_rows": excluded,
                    "exclusions": _canonical_json_text(dict(sorted(cell.exclusions.items()))),
                    "zero_share_match_rows": cell.zero_matches,
                    "one_share_match_rows": cell.one_matches,
                    "multiple_share_match_rows": cell.multiple_matches,
                    "formula_pass_rows": cell.passed,
                    "formula_fail_rows": cell.failed,
                    "ratio_p01": self._quantile(group_key, Decimal("0.01")),
                    "ratio_p05": self._quantile(group_key, Decimal("0.05")),
                    "ratio_p25": self._quantile(group_key, Decimal("0.25")),
                    "ratio_p50": self._quantile(group_key, Decimal("0.50")),
                    "ratio_p75": self._quantile(group_key, Decimal("0.75")),
                    "ratio_p95": self._quantile(group_key, Decimal("0.95")),
                    "ratio_p99": self._quantile(group_key, Decimal("0.99")),
                    "formula_pass_rate": _rate(cell.passed, cell.comparable),
                    "threshold_rule": "full_population_Decimal_ratio_quantiles_are_diagnostic_only_hard_threshold_uses_exact_quantization_overlap",
                    "status": status,
                    "severity": "hard" if hard else "diagnostic",
                }
            )
        return rows


DAILY_SELECTED_COLUMNS = (
    "PERMNO", "DlyCalDt", "PrimaryExch", "ShareType", "DlyPrc", "DlyPrcFlg",
    "DlyCap", "DlyRet", "DlyRetx", "DlyRetMissFlg", "DlyRetDurFlg", "DlyVol",
    "DlyClose", "DlyLow", "DlyHigh", "DlyBid", "DlyAsk", "DlyOpen", "DlyPrcVol",
)
MONTHLY_SELECTED_COLUMNS = (
    "PERMNO", "SecInfoStartDt", "SecInfoEndDt", "YYYYMM", "MthCalDt", "MthPrc",
    "MthPrcFlg", "MthPrcDt", "MthCap", "MthRet", "MthRetx", "MthRetFlg",
    "MthDisCnt", "MthVol", "MthVolFlg", "MthPrcVol", "MthFacShrFlg",
    "MthPrcVolMissCnt", "ShrOut", "PrimaryExch", "ShareType",
)
SHARES_SELECTED_COLUMNS = (
    "PERMNO", "ShrStartDt", "ShrEndDt", "ShrOut", "ShrAdrFlg", "PrimaryExch",
)


def _source_path(settings: NumericUnitSettings, source: str) -> Path:
    if source not in SOURCE_FILENAMES:
        raise NumericUnitProfileError(f"Unknown raw source {source!r}")
    configured = str(settings.config["sources"][source]["filename"])
    candidate = Path(configured)
    if (
        configured != SOURCE_FILENAMES[source]
        or candidate.is_absolute()
        or len(candidate.parts) != 1
        or candidate.name != configured
        or configured in {".", ".."}
        or "/" in configured
        or "\\" in configured
    ):
        raise NumericUnitProfileError(
            f"Configured {source} source is not its exact direct raw filename"
        )
    raw_root = _absolute(settings.raw_root)
    path = _absolute(raw_root / configured)
    if path.parent != raw_root:
        raise NumericUnitProfileError("Raw source escapes raw_root")
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current = current / component
        if current.exists() and current.is_symlink():
            raise NumericUnitProfileError(
                f"Symlink is prohibited in raw source path: {configured}"
            )
    return path


def _check_header(
    handle: io.RawIOBase, source_name: str, expected: Sequence[str]
) -> None:
    """Validate the exact header through the already-authenticated descriptor."""
    try:
        handle.seek(0)
        header_bytes = handle.readline(1 << 20)
        if len(header_bytes) == 1 << 20 and not header_bytes.endswith((b"\n", b"\r")):
            raise NumericUnitProfileError(
                f"CSV header exceeds the bounded limit for {source_name}"
            )
        actual = next(csv.reader([header_bytes.decode("utf-8")]))
        handle.seek(0)
    except (OSError, StopIteration, UnicodeDecodeError, csv.Error) as exc:
        raise NumericUnitProfileError(
            f"Cannot read CSV header {source_name}: {exc}"
        ) from exc
    if actual != list(expected):
        raise NumericUnitProfileError(f"Exact CSV header differs for {source_name}")


def _open_string_reader(
    handle: io.RawIOBase,
    *,
    source_name: str,
    header: Sequence[str],
    selected: Sequence[str],
    block_size_bytes: int,
) -> pacsv.CSVStreamingReader:
    _check_header(handle, source_name, header)
    missing = sorted(set(selected) - set(header))
    if missing:
        raise NumericUnitProfileError(
            f"Selected columns absent in {source_name}: {missing}"
        )
    try:
        return pacsv.open_csv(
            handle,
            read_options=pacsv.ReadOptions(
                block_size=block_size_bytes,
                use_threads=False,
                encoding="utf8",
            ),
            parse_options=pacsv.ParseOptions(delimiter=",", newlines_in_values=False),
            convert_options=pacsv.ConvertOptions(
                column_types={column: pa.string() for column in selected},
                include_columns=list(selected),
                strings_can_be_null=False,
                quoted_strings_can_be_null=False,
                null_values=[],
            ),
        )
    except (OSError, pa.ArrowException) as exc:
        raise NumericUnitProfileError(f"Cannot stream {source_name}: {exc}") from exc


def _batch_rows(batch: pa.RecordBatch) -> Iterator[dict[str, str]]:
    columns = [batch.column(index).to_pylist() for index in range(batch.num_columns)]
    names = batch.schema.names
    for values in zip(*columns, strict=True):
        yield dict(zip(names, values, strict=True))


def _scope_mask(
    batch: pa.RecordBatch,
    *,
    date_column: str,
    start_date: dt.date,
    end_date: dt.date,
    label: str,
) -> tuple[list[bool], list[dt.date]]:
    # This is deliberately the only column inspected before retained filtering.
    raw_dates = batch.column(batch.schema.get_field_index(date_column)).to_pylist()
    parsed_dates = [
        _strict_date(value, f"{label} scope date") for value in raw_dates
    ]
    return [start_date <= value <= end_date for value in parsed_dates], parsed_dates


def _filter_batch(batch: pa.RecordBatch, mask: Sequence[bool]) -> pa.RecordBatch:
    if not any(mask):
        return batch.slice(0, 0)
    return batch.filter(pa.array(mask, type=pa.bool_()))


def _numeric_stratum(source: str, column: str, row: Mapping[str, str]) -> dict[str, str]:
    if source == "daily":
        if column in {"DlyPrc", "DlyCap", "DlyPrcVol"}:
            return {"DlyPrcFlg": row["DlyPrcFlg"]}
        if column in {"DlyRet", "DlyRetx"}:
            return {
                "DlyRetMissFlg": row["DlyRetMissFlg"],
                "DlyRetDurFlg": row["DlyRetDurFlg"],
            }
        return {}
    if source == "monthly":
        if column in {"MthPrc", "MthCap"}:
            return {"MthPrcFlg": row["MthPrcFlg"]}
        if column in {"MthRet", "MthRetx"}:
            return {"MthRetFlg": row["MthRetFlg"]}
        if column in {"MthVol", "MthPrcVol", "MthPrcVolMissCnt"}:
            return {
                "MthVolFlg": row["MthVolFlg"],
                "MthFacShrFlg": row["MthFacShrFlg"],
            }
        return {}
    return {"ShrAdrFlg": row["ShrAdrFlg"]}


def _create_scratch(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA journal_mode=OFF;
        PRAGMA synchronous=OFF;
        PRAGMA temp_store=FILE;
        PRAGMA cache_size=-32768;
        CREATE TABLE shares (
            permno INTEGER NOT NULL,
            start_date TEXT NOT NULL,
            end_date TEXT NOT NULL,
            shrout TEXT NOT NULL,
            adr_flag TEXT NOT NULL,
            primary_exchange TEXT NOT NULL,
            PRIMARY KEY (permno, start_date, end_date)
        ) WITHOUT ROWID;
        CREATE TABLE daily (
            permno INTEGER NOT NULL,
            obs_date TEXT NOT NULL,
            primary_exchange TEXT NOT NULL,
            share_type TEXT NOT NULL,
            price TEXT NOT NULL,
            cap TEXT NOT NULL,
            volume TEXT NOT NULL,
            price_volume TEXT NOT NULL,
            PRIMARY KEY (permno, obs_date)
        ) WITHOUT ROWID;
        CREATE TABLE monthly (
            permno INTEGER NOT NULL,
            cal_date TEXT NOT NULL,
            price_date TEXT NOT NULL,
            primary_exchange TEXT NOT NULL,
            share_type TEXT NOT NULL,
            price TEXT NOT NULL,
            cap TEXT NOT NULL,
            volume TEXT NOT NULL,
            price_volume TEXT NOT NULL,
            price_volume_missing_count TEXT NOT NULL,
            embedded_shrout TEXT NOT NULL,
            volume_flag TEXT NOT NULL,
            share_factor_flag TEXT NOT NULL,
            PRIMARY KEY (permno, cal_date)
        ) WITHOUT ROWID;
        CREATE INDEX shares_lookup ON shares(permno, start_date, end_date);
        CREATE TABLE shares_seen (
            permno INTEGER NOT NULL,
            start_date TEXT NOT NULL,
            end_date TEXT NOT NULL,
            payload_sha256 TEXT NOT NULL,
            PRIMARY KEY (permno, start_date, end_date)
        ) WITHOUT ROWID;
        """
    )


@dataclasses.dataclass
class _AuthenticatedCsvSource:
    opened: _OpenedRegularFile
    handle: io.RawIOBase
    size_bytes: int
    sha256: str

    def __enter__(self) -> _AuthenticatedCsvSource:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: Any,
    ) -> bool:
        try:
            if not self.handle.closed:
                self.handle.close()
        finally:
            self.opened.close(validate=True)
        return False


def _open_authenticated_csv_source(
    settings: NumericUnitSettings, source: str, path: Path
) -> _AuthenticatedCsvSource:
    """Hash and scan one raw CSV through one stable no-follow descriptor."""
    opened = _open_nofollow_regular_file(path, f"raw source {path.name}")
    try:
        size = int(opened.opened.st_size)
        digest = _hash_descriptor(opened.descriptor)
        opened.assert_stable()
        expected = settings.config["sources"][source]
        if settings.authenticate_inputs:
            if size != int(expected["size_bytes"]):
                raise NumericUnitProfileError(
                    f"Authenticated size differs for {path.name}"
                )
            if digest != expected["sha256"]:
                raise NumericUnitProfileError(
                    f"Authenticated SHA-256 differs for {path.name}"
                )
        os.lseek(opened.descriptor, 0, os.SEEK_SET)
        handle = os.fdopen(
            opened.descriptor, "rb", buffering=0, closefd=False
        )
        return _AuthenticatedCsvSource(opened, handle, size, digest)
    except BaseException:
        opened.close(validate=False)
        raise


def _verify_source_authentication(
    settings: NumericUnitSettings, source: str, path: Path
) -> tuple[int, str]:
    """Compatibility helper for authentication-only callers; scanners use one FD."""
    digest, size = _hash_authenticated_regular_file(
        path, f"raw source {path.name}"
    )
    expected = settings.config["sources"][source]
    if settings.authenticate_inputs:
        if size != int(expected["size_bytes"]):
            raise NumericUnitProfileError(f"Authenticated size differs for {path.name}")
        if digest != expected["sha256"]:
            raise NumericUnitProfileError(f"Authenticated SHA-256 differs for {path.name}")
    return size, digest


def _observe_row_numerics(
    inventory: _NumericInventory,
    *,
    source: str,
    date: dt.date,
    row: Mapping[str, str],
    columns: Sequence[str],
) -> dict[str, ClassifiedNumeric]:
    observed: dict[str, ClassifiedNumeric] = {}
    for column in columns:
        observed[column] = inventory.observe(
            source=source,
            column=column,
            date=date,
            stratum=_numeric_stratum(source, column, row),
            raw=row[column],
        )
    return observed


def _identity_key(
    identity_id: str,
    source: str,
    date: dt.date,
    stratum: Mapping[str, Any],
    severity: str,
    tolerance: str,
) -> tuple[Any, ...]:
    return (
        identity_id,
        source,
        date.year,
        "embargo_2014_12" if date.year == 2014 and date.month == 12 else "development_2000_2014_11",
        _json_dimension(stratum),
        severity,
        tolerance,
    )


def _interval_error(left: ParsedNumeric, right: ParsedNumeric) -> tuple[Decimal, Decimal | None]:
    absolute = abs(_exact_subtract(left.value, right.value))
    return absolute, _relative_error(absolute, left.value, right.value)


def _scan_shares(
    settings: NumericUnitSettings,
    connection: sqlite3.Connection,
    numeric: _NumericInventory,
) -> dict[str, Any]:
    source = settings.config["sources"]["shares"]
    path = _source_path(settings, "shares")
    with _open_authenticated_csv_source(settings, "shares", path) as authenticated:
        reader = _open_string_reader(
            authenticated.handle,
            source_name=path.name,
            header=source["header"],
            selected=source["header"],
            block_size_bytes=settings.block_size_bytes,
        )
        try:
            return _scan_shares_from_reader(
                settings,
                connection,
                numeric,
                reader,
                size=authenticated.size_bytes,
                digest=authenticated.sha256,
            )
        finally:
            reader.close()


def _scan_shares_from_reader(
    settings: NumericUnitSettings,
    connection: sqlite3.Connection,
    numeric: _NumericInventory,
    reader: pacsv.CSVStreamingReader,
    *,
    size: int,
    digest: str,
) -> dict[str, Any]:
    source = settings.config["sources"]["shares"]
    scope_start = _strict_date(settings.config["scope"]["start_date"], "scope start")
    scope_end = _strict_date(settings.config["scope"]["quality_end_date"], "scope end")
    full_rows = retained_rows = 0
    for batch in reader:
        full_rows += batch.num_rows
        # Both endpoints are routing fields for an inclusive-overlap source.
        starts_raw = batch.column(batch.schema.get_field_index("ShrStartDt")).to_pylist()
        ends_raw = batch.column(batch.schema.get_field_index("ShrEndDt")).to_pylist()
        starts = [_strict_date(value, "shares interval start") for value in starts_raw]
        ends = [_strict_date(value, "shares interval end") for value in ends_raw]
        if any(start > end for start, end in zip(starts, ends, strict=True)):
            raise NumericUnitProfileError("Shares source contains an inverted interval")
        mask = [
            start <= scope_end and scope_start <= end
            for start, end in zip(starts, ends, strict=True)
        ]
        retained = _filter_batch(batch, mask)
        for row in _batch_rows(retained):
            retained_rows += 1
            permno = _positive_int(row["PERMNO"], "shares PERMNO")
            start = _strict_date(row["ShrStartDt"], "shares retained interval start")
            end = _strict_date(row["ShrEndDt"], "shares retained interval end")
            payload_sha = hashlib.sha256(
                canonical_json_bytes([row[column] for column in source["header"]])
            ).hexdigest()
            existing = connection.execute(
                "SELECT payload_sha256 FROM shares_seen WHERE permno=? AND start_date=? AND end_date=?",
                (permno, start.isoformat(), end.isoformat()),
            ).fetchone()
            if existing is not None:
                if existing[0] != payload_sha:
                    raise NumericUnitProfileError(
                        "Shares duplicate key has conflicting payload"
                    )
                continue
            connection.execute(
                "INSERT INTO shares_seen VALUES (?, ?, ?, ?)",
                (permno, start.isoformat(), end.isoformat(), payload_sha),
            )
            date_for_inventory = max(start, scope_start)
            _observe_row_numerics(
                numeric,
                source="shares",
                date=date_for_inventory,
                row=row,
                columns=("ShrOut",),
            )
            connection.execute(
                "INSERT INTO shares VALUES (?, ?, ?, ?, ?, ?)",
                (
                    permno, start.isoformat(), end.isoformat(), row["ShrOut"],
                    row["ShrAdrFlg"], row["PrimaryExch"],
                ),
            )
    connection.commit()
    overlap = connection.execute(
        """
        SELECT COUNT(*) FROM (
            SELECT start_date,
                   LAG(end_date) OVER (PARTITION BY permno ORDER BY start_date, end_date) AS previous_end
            FROM shares
        ) WHERE previous_end IS NOT NULL AND start_date <= previous_end
        """
    ).fetchone()[0]
    if overlap:
        raise NumericUnitProfileError("Retained shares intervals overlap for a PERMNO")
    if settings.authenticate_inputs:
        if full_rows != int(source["full_file_rows"]):
            raise NumericUnitProfileError("Authenticated shares full row count differs")
        if retained_rows != int(source["verified_retained_rows"]):
            raise NumericUnitProfileError("Verified shares retained row count differs")
    return {
        "source": "shares",
        "filename": source["filename"],
        "size_bytes": size,
        "sha256": digest,
        "full_rows": full_rows,
        "retained_rows": retained_rows,
        "canonical_rows": connection.execute("SELECT COUNT(*) FROM shares").fetchone()[0],
    }


def _scan_daily(
    settings: NumericUnitSettings,
    connection: sqlite3.Connection,
    numeric: _NumericInventory,
    returns: _ReturnInventory,
    identities: _IdentityInventory,
) -> dict[str, Any]:
    source = settings.config["sources"]["daily"]
    path = _source_path(settings, "daily")
    with _open_authenticated_csv_source(settings, "daily", path) as authenticated:
        reader = _open_string_reader(
            authenticated.handle,
            source_name=path.name,
            header=source["header"],
            # Full retained rows are needed only to prove exact duplicate collapse;
            # downstream numeric work still touches the narrow selected contract.
            selected=source["header"],
            block_size_bytes=settings.block_size_bytes,
        )
        try:
            return _scan_daily_from_reader(
                settings,
                connection,
                numeric,
                returns,
                identities,
                reader,
                size=authenticated.size_bytes,
                digest=authenticated.sha256,
            )
        finally:
            reader.close()


def _scan_daily_from_reader(
    settings: NumericUnitSettings,
    connection: sqlite3.Connection,
    numeric: _NumericInventory,
    returns: _ReturnInventory,
    identities: _IdentityInventory,
    reader: pacsv.CSVStreamingReader,
    *,
    size: int,
    digest: str,
) -> dict[str, Any]:
    source = settings.config["sources"]["daily"]
    scope_start = _strict_date(settings.config["scope"]["start_date"], "scope start")
    scope_end = _strict_date(settings.config["scope"]["quality_end_date"], "scope end")
    decimal_columns = list(settings.policy["fields"]["daily"]["decimal_columns"])
    integer_columns = list(
        settings.policy["fields"]["daily"]["nonnegative_integer_columns"]
    )
    full_rows = retained_rows = canonical_rows = 0
    previous_key: tuple[int, dt.date] | None = None
    previous_row: tuple[str, ...] | None = None
    for batch in reader:
        full_rows += batch.num_rows
        mask, _ = _scope_mask(
            batch,
            date_column="DlyCalDt",
            start_date=scope_start,
            end_date=scope_end,
            label="daily",
        )
        retained = _filter_batch(batch, mask)
        for row in _batch_rows(retained):
            retained_rows += 1
            permno = _positive_int(row["PERMNO"], "daily PERMNO")
            date = _strict_date(row["DlyCalDt"], "daily retained date")
            key = (permno, date)
            row_tuple = tuple(row[column] for column in source["header"])
            if previous_key is not None and key < previous_key:
                raise NumericUnitProfileError("Daily canonical key is not nondecreasing")
            if key == previous_key:
                if row_tuple != previous_row:
                    raise NumericUnitProfileError("Daily duplicate key has conflicting payload")
                continue
            previous_key, previous_row = key, row_tuple
            canonical_rows += 1
            classified = _observe_row_numerics(
                numeric,
                source="daily",
                date=date,
                row=row,
                columns=decimal_columns + integer_columns,
            )
            returns.observe_daily(
                date=date,
                flag=row["DlyRetMissFlg"],
                duration=row["DlyRetDurFlg"],
                total=classified["DlyRet"],
                returnx=classified["DlyRetx"],
            )
            identity_key = _identity_key(
                "daily_price_volume",
                "daily",
                date,
                {"DlyPrcFlg": row["DlyPrcFlg"]},
                "hard",
                "quantization_interval_overlap",
            )
            price = classified["DlyPrc"]
            volume = classified["DlyVol"]
            reported = classified["DlyPrcVol"]
            if price.state != "valid_in_domain":
                identities.exclude(identity_key, "DlyPrc_missing_or_invalid")
            elif volume.state != "valid_in_domain":
                identities.exclude(identity_key, "DlyVol_missing_or_invalid")
            elif reported.state != "valid_in_domain":
                identities.compare(identity_key, False)
            else:
                assert price.parsed and volume.parsed and reported.parsed
                calculated_value = _exact_multiply(
                    price.parsed.value, volume.parsed.value
                )
                absolute = abs(
                    _exact_subtract(calculated_value, reported.parsed.value)
                )
                relative = _relative_error(
                    absolute, calculated_value, reported.parsed.value
                )
                identities.compare(
                    identity_key,
                    quantized_product_matches(price.parsed, volume.parsed, reported.parsed),
                    absolute_error=absolute,
                    relative_error=relative,
                )
            connection.execute(
                "INSERT INTO daily VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    permno, date.isoformat(), row["PrimaryExch"], row["ShareType"],
                    row["DlyPrc"], row["DlyCap"], row["DlyVol"], row["DlyPrcVol"],
                ),
            )
    connection.commit()
    if settings.authenticate_inputs:
        if full_rows != int(source["full_file_rows"]):
            raise NumericUnitProfileError("Authenticated daily full row count differs")
        if retained_rows != int(source["verified_retained_rows"]):
            raise NumericUnitProfileError("Verified daily retained row count differs")
        if canonical_rows != int(source["verified_canonical_rows"]):
            raise NumericUnitProfileError("Verified daily canonical row count differs")
    return {
        "source": "daily",
        "filename": source["filename"],
        "size_bytes": size,
        "sha256": digest,
        "full_rows": full_rows,
        "retained_rows": retained_rows,
        "canonical_rows": canonical_rows,
    }


def _select_monthly_group(rows: Sequence[dict[str, str]]) -> dict[str, str]:
    exact_unique: list[dict[str, str]] = []
    seen: set[tuple[str, ...]] = set()
    raw_payloads: dict[tuple[str, str], tuple[str, ...]] = {}
    for row in rows:
        payload = tuple(
            row.get("__full_payload__", tuple(row[column] for column in MONTHLY_SELECTED_COLUMNS))
        )
        raw_key = (row["SecInfoStartDt"], row["SecInfoEndDt"])
        previous = raw_payloads.setdefault(raw_key, payload)
        if previous != payload:
            raise NumericUnitProfileError("Monthly raw key has conflicting payload")
        if payload not in seen:
            seen.add(payload)
            exact_unique.append(row)
    if len(exact_unique) == 1:
        return exact_unique[0]
    price_matches: list[int] = []
    calendar_matches: list[int] = []
    for index, row in enumerate(exact_unique):
        start = _strict_date(row["SecInfoStartDt"], "monthly security interval start")
        end = _strict_date(row["SecInfoEndDt"], "monthly security interval end")
        if start > end:
            raise NumericUnitProfileError("Monthly security interval is inverted")
        calendar = _strict_date(row["MthCalDt"], "monthly calendar date")
        if start <= calendar <= end:
            calendar_matches.append(index)
        if row["MthPrcDt"] != "":
            price_date = _strict_date(row["MthPrcDt"], "monthly price date")
            if start <= price_date <= end:
                price_matches.append(index)
    if (
        len(price_matches) != 1
        or len(calendar_matches) != 1
        or price_matches[0] != calendar_matches[0]
    ):
        raise NumericUnitProfileError(
            "Monthly effective-interval candidate selection is not unique"
        )
    return exact_unique[price_matches[0]]


def _process_monthly_row(
    row: Mapping[str, str],
    *,
    settings: NumericUnitSettings,
    connection: sqlite3.Connection,
    numeric: _NumericInventory,
    returns: _ReturnInventory,
    verified_trading_dates: frozenset[dt.date],
) -> None:
    permno = _positive_int(row["PERMNO"], "monthly PERMNO")
    date = _strict_date(row["MthCalDt"], "monthly retained date")
    if not _fullmatch(r"[0-9]{6}", row["YYYYMM"]):
        raise NumericUnitProfileError("Monthly YYYYMM is not six ASCII digits")
    expected_period = f"{date.year:04d}{date.month:02d}"
    if row["YYYYMM"] != expected_period:
        raise NumericUnitProfileError("Monthly YYYYMM differs from MthCalDt period")
    if row["MthPrcDt"]:
        price_date = _strict_date(row["MthPrcDt"], "monthly observation date")
        if f"{price_date.year:04d}{price_date.month:02d}" != row["YYYYMM"]:
            raise NumericUnitProfileError("MthPrcDt differs from YYYYMM period")
        if price_date not in verified_trading_dates:
            raise NumericUnitProfileError(
                "Nonblank MthPrcDt is absent from the verified trading calendar"
            )
    decimal_columns = list(settings.policy["fields"]["monthly"]["decimal_columns"])
    integer_columns = list(
        settings.policy["fields"]["monthly"]["nonnegative_integer_columns"]
    ) + list(settings.policy["fields"]["monthly"]["positive_integer_when_present_columns"])
    classified = _observe_row_numerics(
        numeric,
        source="monthly",
        date=date,
        row=row,
        columns=decimal_columns + integer_columns,
    )
    returns.observe_monthly(
        date=date,
        flag=row["MthRetFlg"],
        total=classified["MthRet"],
        returnx=classified["MthRetx"],
    )
    connection.execute(
        "INSERT INTO monthly VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            permno, date.isoformat(), row["MthPrcDt"], row["PrimaryExch"],
            row["ShareType"], row["MthPrc"], row["MthCap"], row["MthVol"],
            row["MthPrcVol"], row["MthPrcVolMissCnt"], row["ShrOut"],
            row["MthVolFlg"], row["MthFacShrFlg"],
        ),
    )


def _scan_monthly(
    settings: NumericUnitSettings,
    connection: sqlite3.Connection,
    numeric: _NumericInventory,
    returns: _ReturnInventory,
    verified_trading_dates: frozenset[dt.date] | None = None,
) -> dict[str, Any]:
    source = settings.config["sources"]["monthly"]
    if verified_trading_dates is None:
        verified_trading_dates = frozenset(
            _strict_date(row[0], "synthetic observed trading date")
            for row in connection.execute(
                "SELECT DISTINCT obs_date FROM daily ORDER BY obs_date"
            )
        )
    path = _source_path(settings, "monthly")
    with _open_authenticated_csv_source(settings, "monthly", path) as authenticated:
        reader = _open_string_reader(
            authenticated.handle,
            source_name=path.name,
            header=source["header"],
            selected=source["header"],
            block_size_bytes=settings.block_size_bytes,
        )
        try:
            return _scan_monthly_from_reader(
                settings,
                connection,
                numeric,
                returns,
                verified_trading_dates,
                reader,
                size=authenticated.size_bytes,
                digest=authenticated.sha256,
            )
        finally:
            reader.close()


def _scan_monthly_from_reader(
    settings: NumericUnitSettings,
    connection: sqlite3.Connection,
    numeric: _NumericInventory,
    returns: _ReturnInventory,
    verified_trading_dates: frozenset[dt.date],
    reader: pacsv.CSVStreamingReader,
    *,
    size: int,
    digest: str,
) -> dict[str, Any]:
    source = settings.config["sources"]["monthly"]
    scope_start = _strict_date(settings.config["scope"]["start_date"], "scope start")
    scope_end = _strict_date(settings.config["scope"]["quality_end_date"], "scope end")
    full_rows = retained_rows = canonical_rows = 0
    previous_key: tuple[int, dt.date] | None = None
    group: list[dict[str, str]] = []

    def flush() -> None:
        nonlocal canonical_rows, group
        if not group:
            return
        selected = _select_monthly_group(group)
        _process_monthly_row(
            selected,
            settings=settings,
            connection=connection,
            numeric=numeric,
            returns=returns,
            verified_trading_dates=verified_trading_dates,
        )
        canonical_rows += 1
        group = []

    for batch in reader:
        full_rows += batch.num_rows
        mask, _ = _scope_mask(
            batch,
            date_column="MthCalDt",
            start_date=scope_start,
            end_date=scope_end,
            label="monthly",
        )
        retained = _filter_batch(batch, mask)
        for row in _batch_rows(retained):
            retained_rows += 1
            row["__full_payload__"] = tuple(row[column] for column in source["header"])
            interval_start = _strict_date(
                row["SecInfoStartDt"], "monthly security interval start"
            )
            interval_end = _strict_date(
                row["SecInfoEndDt"], "monthly security interval end"
            )
            if interval_start > interval_end:
                raise NumericUnitProfileError("Monthly security interval is inverted")
            if row["MthPrcDt"]:
                _strict_date(row["MthPrcDt"], "monthly raw observation date")
            key = (
                _positive_int(row["PERMNO"], "monthly PERMNO"),
                _strict_date(row["MthCalDt"], "monthly retained date"),
            )
            if previous_key is not None and key < previous_key:
                raise NumericUnitProfileError("Monthly canonical key is not nondecreasing")
            if previous_key is not None and key != previous_key:
                flush()
            group.append(row)
            previous_key = key
    flush()
    connection.commit()
    if settings.authenticate_inputs:
        if full_rows != int(source["full_file_rows"]):
            raise NumericUnitProfileError("Authenticated monthly full row count differs")
        if retained_rows != int(source["verified_retained_rows"]):
            raise NumericUnitProfileError("Verified monthly retained row count differs")
        if canonical_rows != int(source["verified_canonical_rows"]):
            raise NumericUnitProfileError("Verified monthly canonical row count differs")
    return {
        "source": "monthly",
        "filename": source["filename"],
        "size_bytes": size,
        "sha256": digest,
        "full_rows": full_rows,
        "retained_rows": retained_rows,
        "canonical_rows": canonical_rows,
    }


def _share_matches(
    connection: sqlite3.Connection, permno: int, observation_date: dt.date
) -> list[dict[str, str]]:
    """Return effective share intervals for one key/date (scratch data only)."""
    cursor = connection.execute(
        """
        SELECT start_date, end_date, shrout, adr_flag, primary_exchange
        FROM shares
        WHERE permno = ? AND start_date <= ? AND end_date >= ?
        ORDER BY start_date, end_date, shrout, adr_flag, primary_exchange
        """,
        (permno, observation_date.isoformat(), observation_date.isoformat()),
    )
    return [
        {
            "ShrStartDt": row[0],
            "ShrEndDt": row[1],
            "ShrOut": row[2],
            "ShrAdrFlg": row[3],
            "PrimaryExch": row[4],
        }
        for row in cursor
    ]


def _iter_grouped_join(cursor: sqlite3.Cursor) -> Iterator[list[sqlite3.Row | tuple[Any, ...]]]:
    group: list[sqlite3.Row | tuple[Any, ...]] = []
    previous: tuple[Any, Any] | None = None
    for row in cursor:
        key = (row[0], row[1])
        if previous is not None and key != previous:
            yield group
            group = []
        group.append(row)
        previous = key
    if group:
        yield group


def _build_daily_month_aggregates(
    connection: sqlite3.Connection,
    numeric: _NumericInventory,
) -> None:
    connection.execute(
        """
        CREATE TABLE daily_month (
            permno INTEGER NOT NULL,
            yyyymm TEXT NOT NULL,
            volume_valid INTEGER NOT NULL,
            volume_sum TEXT NOT NULL,
            price_volume_valid INTEGER NOT NULL,
            price_volume_lower TEXT NOT NULL,
            price_volume_upper TEXT NOT NULL,
            price_volume_sum TEXT NOT NULL,
            missing_operand_count INTEGER NOT NULL,
            PRIMARY KEY (permno, yyyymm)
        ) WITHOUT ROWID
        """
    )
    previous: tuple[int, str] | None = None
    volume_valid = True
    volume_sum = Decimal(0)
    price_volume_valid = True
    price_volume_lower = Decimal(0)
    price_volume_upper = Decimal(0)
    price_volume_sum = Decimal(0)
    missing_count = 0

    def flush() -> None:
        if previous is None:
            return
        connection.execute(
            "INSERT INTO daily_month VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                previous[0], previous[1], int(volume_valid), _decimal_text(volume_sum),
                int(price_volume_valid), _decimal_text(price_volume_lower),
                _decimal_text(price_volume_upper), _decimal_text(price_volume_sum),
                missing_count,
            ),
        )

    for permno, obs_date, price_raw, volume_raw, price_volume_raw in connection.execute(
        "SELECT permno, obs_date, price, volume, price_volume FROM daily ORDER BY permno, obs_date"
    ):
        key = (permno, obs_date[:7].replace("-", ""))
        if previous is not None and key != previous:
            flush()
            volume_valid = True
            volume_sum = Decimal(0)
            price_volume_valid = True
            price_volume_lower = Decimal(0)
            price_volume_upper = Decimal(0)
            price_volume_sum = Decimal(0)
            missing_count = 0
        previous = key
        volume = numeric.classify("daily", "DlyVol", volume_raw)
        price = numeric.classify("daily", "DlyPrc", price_raw)
        price_volume = numeric.classify("daily", "DlyPrcVol", price_volume_raw)
        if volume.state == "valid_in_domain":
            assert volume.parsed
            volume_sum = _exact_add(volume_sum, volume.parsed.value)
        else:
            volume_valid = False
        if price_volume.state == "valid_in_domain":
            assert price_volume.parsed
            price_volume_lower = _exact_add(
                price_volume_lower, price_volume.parsed.lower
            )
            price_volume_upper = _exact_add(
                price_volume_upper, price_volume.parsed.upper
            )
            price_volume_sum = _exact_add(
                price_volume_sum, price_volume.parsed.value
            )
        else:
            price_volume_valid = False
        if (
            price.state != "valid_in_domain"
            or volume.state != "valid_in_domain"
            or price_volume.state != "valid_in_domain"
        ):
            missing_count += 1
    flush()
    connection.commit()


def _compare_parsed_identity(
    identities: _IdentityInventory,
    key: tuple[Any, ...],
    left: ClassifiedNumeric,
    right: ClassifiedNumeric,
    *,
    left_reason: str,
    right_reason: str,
) -> None:
    if left.state != "valid_in_domain":
        identities.exclude(key, left_reason)
    elif right.state != "valid_in_domain":
        identities.exclude(key, right_reason)
    else:
        assert left.parsed and right.parsed
        absolute, relative = _interval_error(left.parsed, right.parsed)
        identities.compare(
            key,
            intervals_overlap(
                (left.parsed.lower, left.parsed.upper),
                (right.parsed.lower, right.parsed.upper),
            ),
            absolute_error=absolute,
            relative_error=relative,
        )


def _run_daily_monthly_identities(
    connection: sqlite3.Connection,
    numeric: _NumericInventory,
    identities: _IdentityInventory,
) -> None:
    query = """
        SELECT m.permno, m.cal_date, m.price_date, m.price, m.cap,
               d.permno, d.price, d.cap
        FROM monthly AS m
        LEFT JOIN daily AS d
          ON d.permno = m.permno AND d.obs_date = m.price_date
        ORDER BY m.permno, m.cal_date
    """
    for permno, cal_date_raw, price_date_raw, mprice_raw, mcap_raw, dpermno, dprice_raw, dcap_raw in connection.execute(query):
        date = _strict_date(cal_date_raw, "scratch monthly calendar date")
        for identity_id, monthly_column, daily_column, mraw, draw in (
            ("monthly_daily_price", "MthPrc", "DlyPrc", mprice_raw, dprice_raw),
            ("monthly_daily_cap", "MthCap", "DlyCap", mcap_raw, dcap_raw),
        ):
            key = _identity_key(
                identity_id,
                "monthly_daily",
                date,
                {},
                "hard",
                "quantization_interval_overlap_on_exact_MthPrcDt_mapping",
            )
            if price_date_raw == "":
                identities.exclude(key, "blank_MthPrcDt")
            elif dpermno is None:
                identities.exclude(key, "missing_daily_match")
            else:
                left = numeric.classify("monthly", monthly_column, mraw)
                right = numeric.classify("daily", daily_column, draw)
                _compare_parsed_identity(
                    identities,
                    key,
                    left,
                    right,
                    left_reason="monthly_operand_missing_or_invalid",
                    right_reason="daily_operand_missing_or_invalid",
                )


def _run_monthly_aggregate_identities(
    connection: sqlite3.Connection,
    numeric: _NumericInventory,
    identities: _IdentityInventory,
) -> None:
    query = """
        SELECT m.cal_date, m.volume, m.price_volume, m.price_volume_missing_count,
               m.volume_flag, m.share_factor_flag,
               a.volume_valid, a.volume_sum, a.price_volume_valid,
               a.price_volume_lower, a.price_volume_upper, a.price_volume_sum,
               a.missing_operand_count
        FROM monthly AS m
        LEFT JOIN daily_month AS a
          ON a.permno = m.permno
         AND a.yyyymm = substr(m.cal_date, 1, 4) || substr(m.cal_date, 6, 2)
        ORDER BY m.permno, m.cal_date
    """
    for row in connection.execute(query):
        (
            cal_date_raw, mvol_raw, mpv_raw, mmissing_raw, volume_flag, factor_flag,
            daily_volume_valid, daily_volume_sum, daily_pv_valid, daily_pv_lower,
            daily_pv_upper, daily_pv_sum, daily_missing_count,
        ) = row
        date = _strict_date(cal_date_raw, "scratch monthly aggregate date")
        stratum = {"MthVolFlg": volume_flag, "MthFacShrFlg": factor_flag}
        volume_key = _identity_key(
            "monthly_volume_vs_daily_sum",
            "monthly_daily_aggregate",
            date,
            stratum,
            "diagnostic",
            "exact_integer_point_equality",
        )
        price_volume_key = _identity_key(
            "monthly_price_volume_vs_daily_sum",
            "monthly_daily_aggregate",
            date,
            stratum,
            "diagnostic",
            "summed_quantization_interval_overlap",
        )
        missing_key = _identity_key(
            "monthly_price_volume_missing_count_vs_daily",
            "monthly_daily_aggregate",
            date,
            stratum,
            "diagnostic",
            "exact_integer_point_equality",
        )
        if daily_volume_valid is None:
            identities.exclude(volume_key, "missing_daily_month")
            identities.exclude(price_volume_key, "missing_daily_month")
            identities.exclude(missing_key, "missing_daily_month")
            continue
        mvol = numeric.classify("monthly", "MthVol", mvol_raw)
        if mvol.state != "valid_in_domain":
            identities.exclude(volume_key, "monthly_operand_missing_or_invalid")
        elif not daily_volume_valid:
            identities.exclude(volume_key, "daily_operand_missing_or_invalid")
        else:
            assert mvol.parsed
            right = parse_numeric_lexeme(
                daily_volume_sum,
                grammar=r"[0-9]+",
                interval_kind="point",
            )
            assert right
            absolute, relative = _interval_error(mvol.parsed, right)
            identities.compare(
                volume_key,
                mvol.parsed.value == right.value,
                absolute_error=absolute,
                relative_error=relative,
            )
        mpv = numeric.classify("monthly", "MthPrcVol", mpv_raw)
        if mpv.state != "valid_in_domain":
            identities.exclude(price_volume_key, "monthly_operand_missing_or_invalid")
        elif not daily_pv_valid:
            identities.exclude(price_volume_key, "daily_operand_missing_or_invalid")
        else:
            assert mpv.parsed
            lower = Decimal(daily_pv_lower)
            upper = Decimal(daily_pv_upper)
            central = Decimal(daily_pv_sum)
            absolute = abs(_exact_subtract(mpv.parsed.value, central))
            relative = _relative_error(absolute, mpv.parsed.value, central)
            identities.compare(
                price_volume_key,
                intervals_overlap((mpv.parsed.lower, mpv.parsed.upper), (lower, upper)),
                absolute_error=absolute,
                relative_error=relative,
            )
        mmissing = numeric.classify("monthly", "MthPrcVolMissCnt", mmissing_raw)
        if mmissing.state != "valid_in_domain":
            identities.exclude(missing_key, "monthly_operand_missing_or_invalid")
        else:
            assert mmissing.parsed
            expected = Decimal(daily_missing_count)
            absolute = abs(_exact_subtract(mmissing.parsed.value, expected))
            relative = _relative_error(absolute, mmissing.parsed.value, expected)
            identities.compare(
                missing_key,
                mmissing.parsed.value == expected,
                absolute_error=absolute,
                relative_error=relative,
            )


def _run_capitalization_diagnostics(
    connection: sqlite3.Connection,
    numeric: _NumericInventory,
    identities: _IdentityInventory,
    capitalization: _CapitalizationInventory,
) -> None:
    daily_query = """
        SELECT d.permno, d.obs_date, d.primary_exchange, d.share_type,
               d.price, d.cap, s.start_date, s.shrout, s.adr_flag
        FROM daily AS d
        LEFT JOIN shares AS s
          ON s.permno = d.permno
         AND s.start_date <= d.obs_date
         AND s.end_date >= d.obs_date
        ORDER BY d.permno, d.obs_date, s.start_date, s.end_date
    """
    for group in _iter_grouped_join(connection.execute(daily_query)):
        first = group[0]
        permno, date_raw, exchange, share_type, price_raw, cap_raw = first[:6]
        date = _strict_date(date_raw, "scratch daily capitalization date")
        candidates = [row for row in group if row[6] is not None]
        match_count = len(candidates)
        candidate = candidates[0] if match_count == 1 else None
        adr = candidate[8] if candidate is not None else ""
        shares = (
            numeric.classify("shares", "ShrOut", candidate[7])
            if candidate is not None else None
        )
        capitalization.observe(
            formula_id="daily_interval_cap_price_shares",
            periodicity="daily",
            date=date,
            primary_exchange=exchange,
            share_type=share_type,
            adr_flag=adr,
            permno=permno,
            row_token=f"daily|{permno}|{date_raw}",
            match_count=match_count,
            cap=numeric.classify("daily", "DlyCap", cap_raw),
            price=numeric.classify("daily", "DlyPrc", price_raw),
            shares=shares,
        )

    monthly_query = """
        SELECT m.permno, m.cal_date, m.price_date, m.primary_exchange, m.share_type,
               m.price, m.cap, m.embedded_shrout,
               s.start_date, s.shrout, s.adr_flag
        FROM monthly AS m
        LEFT JOIN shares AS s
          ON m.price_date <> ''
         AND s.permno = m.permno
         AND s.start_date <= m.price_date
         AND s.end_date >= m.price_date
        ORDER BY m.permno, m.cal_date, s.start_date, s.end_date
    """
    for group in _iter_grouped_join(connection.execute(monthly_query)):
        first = group[0]
        (
            permno, cal_date_raw, price_date_raw, exchange, share_type,
            price_raw, cap_raw, embedded_raw,
        ) = first[:8]
        date = _strict_date(cal_date_raw, "scratch monthly capitalization date")
        candidates = [row for row in group if row[8] is not None]
        match_count = len(candidates)
        candidate = candidates[0] if match_count == 1 else None
        adr = candidate[10] if candidate is not None else ""
        embedded = numeric.classify("monthly", "ShrOut", embedded_raw)
        interval = (
            numeric.classify("shares", "ShrOut", candidate[9])
            if candidate is not None else None
        )
        cap = numeric.classify("monthly", "MthCap", cap_raw)
        price = numeric.classify("monthly", "MthPrc", price_raw)
        token = f"monthly|{permno}|{cal_date_raw}"
        capitalization.observe(
            formula_id="monthly_embedded_cap_price_shares",
            periodicity="monthly",
            date=date,
            primary_exchange=exchange,
            share_type=share_type,
            adr_flag=adr,
            permno=permno,
            row_token=f"embedded|{token}",
            match_count=match_count,
            cap=cap,
            price=price,
            shares=embedded,
        )
        capitalization.observe(
            formula_id="monthly_interval_cap_price_shares",
            periodicity="monthly",
            date=date,
            primary_exchange=exchange,
            share_type=share_type,
            adr_flag=adr,
            permno=permno,
            row_token=f"interval|{token}",
            match_count=match_count,
            cap=cap,
            price=price,
            shares=interval,
        )
        share_identity = _identity_key(
            "monthly_embedded_vs_interval_ShrOut",
            "monthly_shares",
            date,
            {"ShrAdrFlg": adr},
            "diagnostic",
            "quantization_interval_overlap",
        )
        if price_date_raw == "":
            identities.exclude(share_identity, "blank_MthPrcDt")
        elif match_count == 0:
            identities.exclude(share_identity, "zero_share_interval_matches")
        elif match_count > 1:
            identities.exclude(share_identity, "multiple_share_interval_matches")
        elif embedded.state != "valid_in_domain":
            identities.exclude(share_identity, "embedded_ShrOut_missing_or_invalid")
        elif interval is None or interval.state != "valid_in_domain":
            identities.exclude(share_identity, "interval_ShrOut_missing_or_invalid")
        else:
            assert embedded.parsed and interval.parsed
            absolute, relative = _interval_error(embedded.parsed, interval.parsed)
            identities.compare(
                share_identity,
                intervals_overlap(
                    (embedded.parsed.lower, embedded.parsed.upper),
                    (interval.parsed.lower, interval.parsed.upper),
                ),
                absolute_error=absolute,
                relative_error=relative,
            )
    connection.commit()


def _authenticate_upstreams(settings: NumericUnitSettings) -> list[dict[str, Any]]:
    if not settings.authenticate_upstreams:
        return []
    config = settings.config
    entries = [
        ("methodology_freeze", config["upstream"]["methodology"]["freeze_manifest"], config["upstream"]["methodology"]["freeze_manifest_sha256"]),
        ("phase_01_freeze", config["upstream"]["phase_01"]["freeze_manifest"], config["upstream"]["phase_01"]["freeze_manifest_sha256"]),
        ("phase_02_config", config["upstream"]["phase_02"]["phase_config"], config["upstream"]["phase_02"]["phase_config_sha256"]),
        ("crsp_backbone_contract", config["upstream"]["phase_02"]["crsp_backbone_contract"], config["upstream"]["phase_02"]["crsp_backbone_contract_sha256"]),
        ("contract_verified_pointer", config["upstream"]["contract"]["verified_pointer"], config["upstream"]["contract"]["verified_pointer_sha256"]),
        ("calendar_verified_pointer", config["upstream"]["calendar"]["verified_pointer"], config["upstream"]["calendar"]["verified_pointer_sha256"]),
        ("daily_monthly_profile_verified_pointer", config["upstream"]["daily_monthly_profile"]["verified_pointer"], config["upstream"]["daily_monthly_profile"]["verified_pointer_sha256"]),
        ("supporting_profile_verified_pointer", config["upstream"]["supporting_profile"]["verified_pointer"], config["upstream"]["supporting_profile"]["verified_pointer_sha256"]),
    ]
    receipts: list[dict[str, Any]] = []
    for snapshot_id, relative, expected_sha in entries:
        path, relative_path = _safe_project_relative_path(
            settings, str(relative), f"upstream {snapshot_id}"
        )
        actual_sha, size = _hash_authenticated_regular_file(
            path, f"upstream {snapshot_id}"
        )
        if actual_sha != expected_sha:
            raise NumericUnitProfileError(
                f"Authenticated upstream SHA differs for {snapshot_id}"
            )
        receipts.append(
            {
                "snapshot_id": snapshot_id,
                "relative_path": relative_path,
                "sha256": actual_sha,
                "size_bytes": size,
            }
        )
    return receipts


def _verified_calendar_dates(settings: NumericUnitSettings) -> frozenset[dt.date]:
    """Authenticate the verified pointer chain and load trading-date membership."""
    pointer_config = settings.config["upstream"]["calendar"]
    pointer_path, _ = _safe_project_relative_path(
        settings, pointer_config["verified_pointer"], "verified calendar pointer"
    )
    pointer_payload, pointer_sha, _ = _read_authenticated_regular_file(
        pointer_path, "verified calendar pointer"
    )
    if pointer_sha != pointer_config["verified_pointer_sha256"]:
        raise NumericUnitProfileError("Verified calendar pointer SHA-256 differs")
    pointer = _json_object_from_bytes(pointer_payload, "verified calendar pointer")
    if (
        pointer.get("status") != pointer_config["required_status"]
        or pointer.get("build_id") != pointer_config["build_id"]
        or pointer.get("verification_id") != pointer_config["verification_id"]
    ):
        raise NumericUnitProfileError("Verified calendar identity/status differs")
    artifact_relative = pointer.get("primary_artifact_path")
    manifest_sha = pointer.get("primary_manifest_sha256")
    if not isinstance(artifact_relative, str) or not isinstance(manifest_sha, str):
        raise NumericUnitProfileError("Verified calendar pointer lacks manifest binding")
    manifest_path, _ = _safe_project_relative_path(
        settings,
        Path(artifact_relative) / "manifest.json",
        "verified calendar manifest",
    )
    manifest_payload, actual_manifest_sha, _ = _read_authenticated_regular_file(
        manifest_path, "verified calendar manifest"
    )
    if actual_manifest_sha != manifest_sha:
        raise NumericUnitProfileError("Verified calendar manifest SHA-256 differs")
    manifest = _json_object_from_bytes(
        manifest_payload, "verified calendar manifest"
    )
    matches = [
        item
        for item in manifest.get("outputs", [])
        if isinstance(item, dict)
        and Path(str(item.get("path", ""))).name == "trading_calendar.csv"
    ]
    if len(matches) != 1:
        raise NumericUnitProfileError(
            "Verified calendar manifest lacks exactly one trading_calendar.csv"
        )
    receipt = matches[0]
    calendar_path = _absolute(str(receipt["path"]))
    try:
        calendar_path.relative_to(_absolute(settings.database_root))
    except ValueError as exc:
        raise NumericUnitProfileError(
            "Verified trading calendar is outside database_root"
        ) from exc
    calendar_payload, calendar_sha, calendar_size = _read_authenticated_regular_file(
        calendar_path, "verified trading calendar"
    )
    if calendar_size != int(receipt["size_bytes"]):
        raise NumericUnitProfileError("Verified trading calendar size differs")
    if calendar_sha != receipt["sha256"]:
        raise NumericUnitProfileError("Verified trading calendar SHA-256 differs")
    dates: set[dt.date] = set()
    try:
        text = calendar_payload.decode("utf-8")
        reader = csv.DictReader(io.StringIO(text, newline=""))
        candidates = [
            field
            for field in (reader.fieldnames or [])
            if field in {"trading_date", "observed_date", "calendar_date", "date"}
        ]
        if len(candidates) != 1:
            raise NumericUnitProfileError(
                "Verified trading calendar lacks one recognized date column"
            )
        column = candidates[0]
        for row in reader:
            value = _strict_date(row[column], "verified trading calendar date")
            if value in dates:
                raise NumericUnitProfileError(
                    "Verified trading calendar contains a duplicate date"
                )
            dates.add(value)
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise NumericUnitProfileError(
            f"Cannot read verified trading calendar: {exc}"
        ) from exc
    if not dates:
        raise NumericUnitProfileError("Verified trading calendar is empty")
    return frozenset(dates)


def snapshot_inventory_sha256(
    descriptors: Sequence[Mapping[str, Any]],
) -> str:
    """Hash portable snapshot descriptors with the publisher's exact domain."""
    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_locations: set[tuple[str, str]] = set()
    for descriptor in descriptors:
        required = {
            "relative_path", "root", "sha256", "size_bytes", "snapshot_id"
        }
        if set(descriptor) != required:
            raise NumericUnitProfileError("Snapshot descriptor schema differs")
        snapshot_id = str(descriptor["snapshot_id"])
        root = str(descriptor["root"])
        relative = str(descriptor["relative_path"])
        digest = str(descriptor["sha256"])
        size = descriptor["size_bytes"]
        if not snapshot_id or any(ord(character) < 32 for character in snapshot_id):
            raise NumericUnitProfileError("Snapshot id is invalid")
        if snapshot_id in seen_ids:
            raise NumericUnitProfileError("Snapshot id is duplicated")
        seen_ids.add(snapshot_id)
        if root not in {"project", "raw"}:
            raise NumericUnitProfileError("Snapshot root must be project or raw")
        path = Path(relative)
        if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts) or "\\" in relative:
            raise NumericUnitProfileError("Snapshot relative path is unsafe")
        location = (root, path.as_posix())
        if location in seen_locations:
            raise NumericUnitProfileError("Snapshot location is duplicated")
        seen_locations.add(location)
        if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise NumericUnitProfileError("Snapshot SHA-256 is malformed")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise NumericUnitProfileError("Snapshot size is invalid")
        normalized.append(
            {
                "relative_path": path.as_posix(),
                "root": root,
                "sha256": digest,
                "size_bytes": size,
                "snapshot_id": snapshot_id,
            }
        )
    normalized.sort(key=lambda item: item["snapshot_id"])
    return hashlib.sha256(
        SNAPSHOT_INVENTORY_DOMAIN + canonical_json_bytes(normalized)
    ).hexdigest()


def _portable_snapshot_descriptors(
    settings: NumericUnitSettings,
    source_receipts: Sequence[Mapping[str, Any]],
    upstream_receipts: Sequence[Mapping[str, Any]],
    *,
    policy_sha256: str,
    policy_size_bytes: int,
) -> list[dict[str, Any]]:
    descriptors = [
        {
            "snapshot_id": "policy",
            "root": "project",
            "relative_path": str(settings.config["policy"]["path"]),
            "sha256": policy_sha256,
            "size_bytes": policy_size_bytes,
        }
    ]
    if "empirical_lock" in settings.config and settings.authenticate_upstreams:
        empirical = settings.config["empirical_lock"]
        relative_text = str(empirical["path"])
        relative = Path(relative_text)
        if (
            relative.is_absolute()
            or not relative.parts
            or any(part in {"", ".", ".."} for part in relative.parts)
            or "\\" in relative_text
        ):
            raise NumericUnitProfileError("Empirical-lock snapshot path is unsafe")
        empirical_path = _absolute(settings.project_root / relative)
        expected_sha = str(empirical["sha256"])
        actual_sha, empirical_size = _hash_authenticated_regular_file(
            empirical_path, "empirical-lock snapshot"
        )
        if actual_sha != expected_sha:
            raise NumericUnitProfileError("Empirical-lock snapshot SHA differs")
        descriptors.append(
            {
                "snapshot_id": "empirical_lock",
                "root": "project",
                "relative_path": relative.as_posix(),
                "sha256": actual_sha,
                "size_bytes": empirical_size,
            }
        )
    descriptors.extend(
        {
            "snapshot_id": receipt["snapshot_id"],
            "root": "project",
            "relative_path": receipt["relative_path"],
            "sha256": receipt["sha256"],
            "size_bytes": int(receipt["size_bytes"]),
        }
        for receipt in upstream_receipts
    )
    raw_ids = {"daily": "raw_daily", "monthly": "raw_monthly", "shares": "raw_shares"}
    descriptors.extend(
        {
            "snapshot_id": raw_ids[str(receipt["source"])],
            "root": "raw",
            "relative_path": receipt["filename"],
            "sha256": receipt["sha256"],
            "size_bytes": int(receipt["size_bytes"]),
        }
        for receipt in source_receipts
    )
    return sorted(descriptors, key=lambda item: item["snapshot_id"])


def _sort_rows(
    rows: Iterable[Mapping[str, Any]], sort_fields: Sequence[str]
) -> list[dict[str, Any]]:
    materialized = [dict(row) for row in rows]

    def component(value: Any) -> tuple[int, Any]:
        if isinstance(value, int):
            return (0, value)
        return (1, str(value).encode("utf-8"))

    return sorted(
        materialized,
        key=lambda row: tuple(component(row[field]) for field in sort_fields),
    )


def _semantic_build_id(mode: str, semantic_seed: Mapping[str, Any]) -> str:
    """Bind every pre-ID semantic, including implementation/snapshot bindings."""
    if mode not in {"discovery", "production"}:
        raise NumericUnitProfileError(f"Unknown numeric-unit build mode {mode!r}")
    digest = hashlib.sha256(canonical_json_bytes(semantic_seed)).hexdigest()
    prefix = "crsp-numeric-" if mode == "production" else "crsp-numeric-discovery-"
    return prefix + digest[:24]


def _validate_table_rows(
    table: str,
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
) -> None:
    expected = set(fields)
    for row in rows:
        if set(row) != expected:
            raise NumericUnitProfileError(f"{table} row schema differs")
        if table == "numeric_inventory":
            states = sum(
                int(row[name])
                for name in (
                    "blank_rows", "valid_in_domain_rows", "valid_out_of_domain_rows",
                    "malformed_rows", "nonfinite_rows", "overflow_rows",
                )
            )
            if int(row["population_rows"]) != states:
                raise NumericUnitProfileError("Numeric inventory balance failed")
            if int(row["negative_zero_rows"]) > int(row["valid_in_domain_rows"]) + int(row["valid_out_of_domain_rows"]):
                raise NumericUnitProfileError("Negative-zero subset balance failed")
        elif table == "return_inventory":
            if row["coherent_rows"] != "":
                if int(row["population_rows"]) != int(row["coherent_rows"]) + int(row["incoherent_rows"]):
                    raise NumericUnitProfileError("Return inventory balance failed")
        elif table == "identity_inventory":
            exclusions = json.loads(str(row["exclusions"]))
            if int(row["population_rows"]) != int(row["comparable_rows"]) + int(row["excluded_rows"]):
                raise NumericUnitProfileError("Identity population balance failed")
            if int(row["comparable_rows"]) != int(row["pass_rows"]) + int(row["fail_rows"]):
                raise NumericUnitProfileError("Identity comparable balance failed")
            if int(row["excluded_rows"]) != sum(int(value) for value in exclusions.values()):
                raise NumericUnitProfileError("Identity exclusion balance failed")
        elif table == "capitalization_diagnostics":
            exclusions = json.loads(str(row["exclusions"]))
            if int(row["population_rows"]) != int(row["comparable_rows"]) + int(row["excluded_rows"]):
                raise NumericUnitProfileError("Capitalization population balance failed")
            if int(row["comparable_rows"]) != int(row["formula_pass_rows"]) + int(row["formula_fail_rows"]):
                raise NumericUnitProfileError("Capitalization comparable balance failed")
            if int(row["excluded_rows"]) != sum(int(value) for value in exclusions.values()):
                raise NumericUnitProfileError("Capitalization exclusion balance failed")


def _table_rows_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(canonical_json_bytes(list(rows))).hexdigest()


def _measurement_projection(
    table: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    period_segment: str | None = None,
) -> dict[str, Any]:
    if table not in MEASUREMENT_EXCLUDED_FIELDS:
        raise NumericUnitProfileError(f"Unknown empirical projection table {table!r}")
    excluded = MEASUREMENT_EXCLUDED_FIELDS[table]
    projected = [
        {key: value for key, value in row.items() if key not in excluded}
        for row in rows
        if period_segment is None or row["period_segment"] == period_segment
    ]
    return {
        "row_count": len(projected),
        "sha256": hashlib.sha256(canonical_json_bytes(projected)).hexdigest(),
    }


def _validated_production_expectations(
    settings: NumericUnitSettings,
) -> Mapping[str, Any]:
    binding = settings.config.get("empirical_lock")
    required_binding = {"path", "sha256", "lock_id", "schema_version"}
    if not isinstance(binding, dict) or set(binding) != required_binding:
        raise NumericUnitProfileError("Production empirical-lock binding differs")
    if binding["schema_version"] != 1:
        raise NumericUnitProfileError("Production empirical-lock schema version differs")
    relative_text = str(binding["path"])
    relative = Path(relative_text)
    if (
        relative.is_absolute()
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
        or "\\" in relative_text
    ):
        raise NumericUnitProfileError("Production empirical-lock path is unsafe")
    project_root = _absolute(settings.project_root)
    path = _absolute(project_root / relative)
    try:
        path.relative_to(project_root)
    except ValueError as exc:
        raise NumericUnitProfileError(
            "Production empirical-lock path escapes project_root"
        ) from exc
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current = current / component
        if current.exists() and current.is_symlink():
            raise NumericUnitProfileError(
                "Symlink is prohibited in the empirical-lock path"
            )
    if not path.is_file():
        raise NumericUnitProfileError("Production empirical-lock file is absent")
    expected_sha = str(binding["sha256"])
    if re.fullmatch(r"[0-9a-f]{64}", expected_sha) is None:
        raise NumericUnitProfileError("Production empirical-lock SHA is malformed")
    lock_payload, actual_sha, _ = _read_authenticated_regular_file(
        path, "production empirical lock"
    )
    if actual_sha != expected_sha:
        raise NumericUnitProfileError("Production empirical-lock SHA differs")
    lock = _json_object_from_bytes(lock_payload, "production empirical lock")
    required_top = {
        "schema_version",
        "lock_id",
        "evidence_origin",
        "scope",
        "measurement_projection_contract",
        "measurement_projections",
        "source_expectations",
        "theory_first_rules",
        "transparent_aggregates",
        "exclusion_contract",
        "first_discovery_false_positive_scope_correction",
        "downstream",
    }
    if set(lock) != required_top:
        raise NumericUnitProfileError("Production empirical-lock key set differs")
    if (
        lock["schema_version"] != binding["schema_version"]
        or lock["lock_id"] != binding["lock_id"]
    ):
        raise NumericUnitProfileError("Production empirical-lock identity differs")

    scope = lock["scope"]
    segments = list(settings.config["outputs"]["vocabularies"]["period_segment"])
    if (
        not isinstance(scope, dict)
        or scope.get("validation_or_final_test_rows_permitted") is not False
        or scope.get("date_bounds_are_inclusive") is not True
        or set(scope.get("segments", {})) != set(segments)
    ):
        raise NumericUnitProfileError("Production empirical-lock scope differs")
    configured_scope = settings.config["scope"]
    development = scope["segments"]["development_2000_2014_11"]
    embargo = scope["segments"]["embargo_2014_12"]
    if (
        development.get("start_date") != configured_scope["start_date"]
        or development.get("end_date") != configured_scope["development_end_date"]
        or development.get("may_contribute_to_model_development") is not True
        or embargo.get("start_date") != "2014-12-01"
        or embargo.get("end_date") != configured_scope["quality_end_date"]
        or embargo.get("may_contribute_to_model_development") is not False
        or embargo.get("may_rescue_a_failed_development_threshold") is not False
    ):
        raise NumericUnitProfileError(
            "Production empirical-lock development/embargo boundary differs"
        )

    contract = lock["measurement_projection_contract"]
    if (
        not isinstance(contract, dict)
        or contract.get("algorithm_id")
        != "sha256_canonical_json_lf_projected_rows_v1"
        or contract.get("checks_table_is_not_an_empirical_projection") is not True
        or contract.get("rows_are_sorted_by_profile_output_sort_keys_before_projection")
        is not True
        or contract.get("cell_values_are_preserved_without_reparsing") is not True
    ):
        raise NumericUnitProfileError(
            "Production empirical measurement-projection contract differs"
        )
    projections = lock["measurement_projections"]
    if not isinstance(projections, dict) or set(projections) != set(
        EMPIRICAL_TABLE_NAMES
    ):
        raise NumericUnitProfileError(
            "Production empirical measurement-projection set differs"
        )
    for table, expected in projections.items():
        if not isinstance(expected, dict) or set(expected) != {
            "policy_derived_fields_excluded", "row_count", "segments", "sha256"
        }:
            raise NumericUnitProfileError(
                f"Production empirical projection schema differs for {table}"
            )
        if expected["policy_derived_fields_excluded"] != sorted(
            MEASUREMENT_EXCLUDED_FIELDS[table]
        ):
            raise NumericUnitProfileError(
                f"Production empirical projection fields differ for {table}"
            )
        if set(expected["segments"]) != set(segments):
            raise NumericUnitProfileError(
                f"Production empirical projection segments differ for {table}"
            )
        projections_to_validate = [expected, *expected["segments"].values()]
        if any(
            not isinstance(item.get("row_count"), int)
            or isinstance(item.get("row_count"), bool)
            or item["row_count"] < 0
            or re.fullmatch(r"[0-9a-f]{64}", str(item.get("sha256", ""))) is None
            for item in projections_to_validate
        ):
            raise NumericUnitProfileError(
                f"Production empirical projection value differs for {table}"
            )
        if expected["row_count"] != sum(
            item["row_count"] for item in expected["segments"].values()
        ):
            raise NumericUnitProfileError(
                f"Production empirical projection row balance differs for {table}"
            )

    cap_locked = lock["transparent_aggregates"]["capitalization"][
        "non_ADR_NS_N_primary_threshold"
    ]
    selector = cap_locked["selector"]
    required_selector = {
        "ShareType": "NS",
        "effective_interval_ShrAdrFlg": "N",
        "interval_date": "MthPrcDt",
        "interval_match_count": 1,
        "primary_exchange_scope": "all_observed_no_exchange_filter",
        "required_valid_operands": ["MthCap", "MthPrc", "interval_ShrOut"],
    }
    if selector != required_selector:
        raise NumericUnitProfileError(
            "Production non-ADR capitalization selector differs"
        )
    if (
        cap_locked.get("authoritative_formula_id")
        != "monthly_interval_cap_price_shares"
        or cap_locked.get("required_fail_rows") != 0
        or cap_locked.get("required_pass_rate") != "1"
        or cap_locked.get("development_threshold_must_be_evaluated_separately")
        is not True
    ):
        raise NumericUnitProfileError(
            "Production non-ADR capitalization theory threshold differs"
        )
    segment_counts: dict[str, dict[str, int]] = {}
    if set(cap_locked["segments"]) != set(segments):
        raise NumericUnitProfileError(
            "Production non-ADR capitalization segment set differs"
        )
    for segment in segments:
        source_counts = cap_locked["segments"][segment]
        if set(source_counts) != set(NON_ADR_COUNT_FIELDS):
            raise NumericUnitProfileError(
                f"Production non-ADR source-count schema differs for {segment}"
            )
        segment_counts[segment] = {
            field: int(source_counts[field]) for field in NON_ADR_COUNT_FIELDS
        }
        counts = segment_counts[segment]
        if (
            counts["population_rows"]
            != counts["comparable_rows"] + counts["excluded_rows"]
            or counts["comparable_rows"]
            != counts["formula_pass_rows"] + counts["formula_fail_rows"]
            or counts["formula_fail_rows"] != 0
        ):
            raise NumericUnitProfileError(
                f"Production non-ADR count balance differs for {segment}"
            )
    return {
        "lock": lock,
        "lock_path": relative.as_posix(),
        "lock_sha256": expected_sha,
        "measurement_projections": projections,
        "non_adr_capitalization": {
            "formula_id": "monthly_interval_cap_price_shares",
            "periodicity": "monthly",
            "share_type": "NS",
            "shr_adr_flag": "N",
            "primary_exchange_scope": "all_observed_no_exchange_filter",
            "pass_rule": "quantization_interval_overlap",
            "required_formula_pass_rate": "1",
            "required_formula_fail_rows": 0,
            "required_multiple_share_match_rows": 0,
            "segment_counts": segment_counts,
        },
    }


def _non_adr_capitalization_segment_counts(
    rows: Sequence[Mapping[str, Any]], cap: Mapping[str, Any]
) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for row in rows:
        if any(
            row[field] != cap[field]
            for field in (
                "formula_id", "periodicity", "share_type", "shr_adr_flag"
            )
        ):
            continue
        segment = str(row["period_segment"])
        cell = counts.setdefault(
            segment, {field: 0 for field in NON_ADR_COUNT_FIELDS}
        )
        for field in NON_ADR_COUNT_FIELDS:
            cell[field] += int(row[field])
    return {segment: counts[segment] for segment in sorted(counts)}


def _make_checks(
    settings: NumericUnitSettings,
    source_receipts: Sequence[Mapping[str, Any]],
    tables: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    mode: str,
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []

    def add(
        check_id: str,
        *,
        stage: str,
        family: str,
        source: str,
        severity: str,
        expected: Any,
        actual: Any,
        status: str,
        consequence: str,
        stratum: Mapping[str, Any] | None = None,
        tolerance: str = "exact",
    ) -> None:
        checks.append(
            {
                "check_id": check_id,
                "stage": stage,
                "family": family,
                "source": source,
                "stratum": _json_dimension(stratum),
                "severity": severity,
                "expected": _canonical_json_text(expected),
                "actual": _canonical_json_text(actual),
                "tolerance_rule": tolerance,
                "status": status,
                "downstream_consequence": consequence,
            }
        )

    for receipt in source_receipts:
        source = str(receipt["source"])
        configured = settings.config["sources"][source]
        for metric, expected_field in (
            ("full_rows", "full_file_rows"),
            ("retained_rows", "verified_retained_rows"),
            ("canonical_rows", "verified_canonical_rows"),
        ):
            if expected_field not in configured:
                continue
            actual = int(receipt[metric])
            expected = int(configured[expected_field])
            authenticated = settings.authenticate_inputs
            add(
                f"source_{source}_{metric}",
                stage="source_scan",
                family="authenticated_count",
                source=source,
                severity="hard" if authenticated else "diagnostic",
                expected=expected if authenticated else "authentication_disabled",
                actual=actual,
                status="PASS" if authenticated and actual == expected else "DIAGNOSTIC" if not authenticated else "FAIL",
                consequence="blocks_numeric_unit_profile" if authenticated else "test_or_discovery_only",
            )

    for table, row_group in tables.items():
        failures = sum(row["status"] == "FAIL" for row in row_group)
        add(
            f"{table}_hard_failures",
            stage="aggregate_evidence",
            family="profile_gate",
            source="all",
            severity="hard",
            expected=0,
            actual=failures,
            status="PASS" if failures == 0 else "FAIL",
            consequence="blocks_locked_primary_profile",
        )
    zero_comparable_segments = _identity_zero_comparable_active_segments(
        tables["identity_inventory"]
    )
    add(
        "identity_pooled_active_segment_zero_comparable_groups",
        stage="aggregate_evidence",
        family="identity_coverage",
        source="all",
        severity="hard",
        expected=[],
        actual=zero_comparable_segments,
        status="PASS" if not zero_comparable_segments else "FAIL",
        consequence="blocks_locked_primary_profile",
        stratum={"scope": "identity_source_year_period_segment_across_strata"},
        tolerance="at_least_one_comparable_row_per_active_segment",
    )
    identity_population_differences = _identity_active_segment_population_differences(
        tables["numeric_inventory"], tables["identity_inventory"]
    )
    add(
        "identity_active_segment_population_conservation",
        stage="aggregate_evidence",
        family="identity_coverage",
        source="all",
        severity="hard",
        expected=[],
        actual=identity_population_differences,
        status="PASS" if not identity_population_differences else "FAIL",
        consequence="blocks_locked_primary_profile",
        stratum={"scope": "identity_population_vs_canonical_numeric_grain"},
        tolerance="exact",
    )
    if mode == "production":
        expectations = _validated_production_expectations(settings)
        for table in EMPIRICAL_TABLE_NAMES:
            expected_projection = expectations["measurement_projections"][table]
            actual_projection = _measurement_projection(table, tables[table])
            add(
                f"empirical_{table}_measurement_projection_full",
                stage="locked_empirical_reproduction",
                family="measurement_projection",
                source="all",
                severity="hard",
                expected={
                    "row_count": expected_projection["row_count"],
                    "sha256": expected_projection["sha256"],
                },
                actual=actual_projection,
                status=(
                    "PASS"
                    if actual_projection
                    == {
                        "row_count": expected_projection["row_count"],
                        "sha256": expected_projection["sha256"],
                    }
                    else "FAIL"
                ),
                consequence="blocks_locked_primary_profile",
                stratum={"period_segment": "all_locked_segments"},
                tolerance="exact_sha256_canonical_json_lf_projected_rows_v1",
            )
            for segment, expected_segment in expected_projection["segments"].items():
                actual_segment = _measurement_projection(
                    table, tables[table], period_segment=segment
                )
                add(
                    f"empirical_{table}_measurement_projection_{segment}",
                    stage="locked_empirical_reproduction",
                    family="measurement_projection",
                    source="all",
                    severity="hard",
                    expected=expected_segment,
                    actual=actual_segment,
                    status=(
                        "PASS" if actual_segment == expected_segment else "FAIL"
                    ),
                    consequence="blocks_locked_primary_profile",
                    stratum={"period_segment": segment},
                    tolerance="exact_sha256_canonical_json_lf_projected_rows_v1",
                )

        cap = expectations["non_adr_capitalization"]
        actual_segment_counts = _non_adr_capitalization_segment_counts(
            tables["capitalization_diagnostics"], cap
        )
        expected_segment_counts = cap["segment_counts"]
        add(
            "non_adr_capitalization_segment_counts",
            stage="locked_empirical_reproduction",
            family="capitalization_scale",
            source="monthly",
            severity="hard",
            expected=expected_segment_counts,
            actual=actual_segment_counts,
            status=(
                "PASS"
                if actual_segment_counts == expected_segment_counts
                else "FAIL"
            ),
            consequence="blocks_typed_capitalization_units",
            stratum={
                "formula_id": cap["formula_id"],
                "share_type": cap["share_type"],
                "shr_adr_flag": cap["shr_adr_flag"],
                "primary_exchange_scope": cap["primary_exchange_scope"],
            },
            tolerance="exact_counts",
        )
        totals = {
            field: sum(
                segment[field] for segment in actual_segment_counts.values()
            )
            for field in NON_ADR_COUNT_FIELDS
        }
        actual_theory_rule = {
            "formula_pass_rate": _rate(
                totals["formula_pass_rows"], totals["comparable_rows"]
            ),
            "formula_fail_rows": totals["formula_fail_rows"],
            "multiple_share_match_rows": totals["multiple_share_match_rows"],
            "pass_rule": cap["pass_rule"],
        }
        expected_theory_rule = {
            "formula_pass_rate": cap["required_formula_pass_rate"],
            "formula_fail_rows": cap["required_formula_fail_rows"],
            "multiple_share_match_rows": cap[
                "required_multiple_share_match_rows"
            ],
            "pass_rule": cap["pass_rule"],
        }
        add(
            "non_adr_capitalization_theory_threshold",
            stage="locked_empirical_reproduction",
            family="capitalization_scale",
            source="monthly",
            severity="hard",
            expected=expected_theory_rule,
            actual=actual_theory_rule,
            status=(
                "PASS" if actual_theory_rule == expected_theory_rule else "FAIL"
            ),
            consequence="blocks_typed_capitalization_units",
            stratum={
                "formula_id": cap["formula_id"],
                "share_type": cap["share_type"],
                "shr_adr_flag": cap["shr_adr_flag"],
            },
            tolerance="all_comparable_quantization_intervals_must_overlap",
        )
    expected_profile_status = (
        "discovery_only_unlocked"
        if mode == "discovery"
        else settings.config["outputs"]["production_publication_requires_status"]
    )
    add(
        f"{mode}_profile_status",
        stage="gate_separation",
        family="authorization",
        source="all",
        severity="hard",
        expected=expected_profile_status,
        actual=settings.config["status"],
        status="PASS" if settings.config["status"] == expected_profile_status else "FAIL",
        consequence=(
            "all_downstream_actions_remain_blocked"
            if mode == "discovery"
            else "blocks_primary_publication"
        ),
    )
    return checks


def _run_numeric_unit_scan(
    settings: NumericUnitSettings, *, mode: str
) -> dict[str, Any]:
    if mode not in {"discovery", "production"}:
        raise NumericUnitProfileError(f"Unknown numeric-unit scan mode {mode!r}")
    if mode == "production":
        _require_production_locks(settings)
    settings_file_receipts = _bound_settings_file_receipts(settings)
    upstream_receipts = _authenticate_upstreams(settings)
    with tempfile.TemporaryDirectory(prefix="crsp-numeric-unit-") as scratch:
        database_path = Path(scratch) / "numeric_unit.sqlite3"
        connection = sqlite3.connect(database_path)
        try:
            _create_scratch(connection)
            numeric = _NumericInventory(settings.policy)
            returns = _ReturnInventory(settings.policy)
            identities = _IdentityInventory()
            capitalization = _CapitalizationInventory(connection)
            shares_receipt = _scan_shares(settings, connection, numeric)
            daily_receipt = _scan_daily(
                settings, connection, numeric, returns, identities
            )
            if settings.authenticate_upstreams:
                trading_dates = _verified_calendar_dates(settings)
            elif settings.verified_trading_dates is not None:
                trading_dates = settings.verified_trading_dates
            else:
                # Test-only fallback when upstream authentication is explicitly
                # disabled: derive market-wide observed dates, never MthCalDt.
                trading_dates = frozenset(
                    _strict_date(row[0], "synthetic observed trading date")
                    for row in connection.execute(
                        "SELECT DISTINCT obs_date FROM daily ORDER BY obs_date"
                    )
                )
            monthly_receipt = _scan_monthly(
                settings, connection, numeric, returns, trading_dates
            )
            source_receipts = [shares_receipt, daily_receipt, monthly_receipt]
            _build_daily_month_aggregates(connection, numeric)
            _run_daily_monthly_identities(connection, numeric, identities)
            _run_monthly_aggregate_identities(connection, numeric, identities)
            _run_capitalization_diagnostics(
                connection, numeric, identities, capitalization
            )
            raw_tables: dict[str, list[dict[str, Any]]] = {
                "numeric_inventory": numeric.rows(),
                "return_inventory": returns.rows(),
                "identity_inventory": identities.rows(),
                "capitalization_diagnostics": capitalization.rows(),
            }
        finally:
            connection.close()

    schemas = settings.config["outputs"]["schemas"]
    sort_keys = settings.config["outputs"]["sort_keys"]
    tables: dict[str, list[dict[str, Any]]] = {}
    for table, rows in raw_tables.items():
        sorted_rows = _sort_rows(rows, sort_keys[table])
        _validate_table_rows(table, sorted_rows, schemas[table])
        tables[table] = sorted_rows
    checks = _sort_rows(
        _make_checks(settings, source_receipts, tables, mode=mode), sort_keys["checks"]
    )
    _validate_table_rows("checks", checks, schemas["checks"])
    tables["checks"] = checks

    if _bound_settings_file_receipts(settings) != settings_file_receipts:
        raise NumericUnitProfileError(
            "Numeric-unit settings files changed during the scan"
        )
    policy_sha, policy_size = settings_file_receipts["policy"]
    profile_sha, _ = settings_file_receipts["profile"]
    empirical_lock_sha = str(settings.config["empirical_lock"]["sha256"])
    implementation_sha = sha256_file(Path(__file__))
    snapshot_descriptors = _portable_snapshot_descriptors(
        settings,
        source_receipts,
        upstream_receipts,
        policy_sha256=policy_sha,
        policy_size_bytes=policy_size,
    )
    snapshot_inventory_sha = snapshot_inventory_sha256(snapshot_descriptors)
    input_receipt = {
        "schema_version": 1,
        "authentication": {
            "raw_inputs": settings.authenticate_inputs,
            "upstreams": settings.authenticate_upstreams,
        },
        "raw_sources": sorted(source_receipts, key=lambda row: row["source"]),
        "upstreams": sorted(upstream_receipts, key=lambda row: row["snapshot_id"]),
        "portable_snapshots": snapshot_descriptors,
    }
    profile_contract = {
        "schema_version": 1,
        "profile_name": settings.config["profile_name"],
        "profile_version": settings.config["profile_version"],
        "profile_status": settings.config["status"],
        "policy_sha256": policy_sha,
        "empirical_lock_sha256": empirical_lock_sha,
        "profile_config_sha256": profile_sha,
        "scope": {
            "start_date": settings.config["scope"]["start_date"],
            "quality_end_date": settings.config["scope"]["quality_end_date"],
            "december_2014_is_quality_only": True,
        },
        "joins": {
            "daily_shares": "PERMNO plus inclusive DlyCalDt share interval",
            "monthly_shares": "PERMNO plus inclusive nonblank MthPrcDt share interval",
            "monthly_daily": "monthly (PERMNO,MthPrcDt) equals daily (PERMNO,DlyCalDt)",
            "MthCalDt_substitution_permitted": False,
        },
        "capitalization_formula_ids": list(
            settings.config["outputs"]["vocabularies"]["capitalization_formula_id"]
        ),
        "capitalization_quantiles": {
            "estimator": "full_population_type7_linear_interpolation_over_disk_backed_Decimal_ratios_with_at_least_50_significant_digits",
            "sampled": False,
            "hard_threshold_use": False,
        },
        "daily_price_flag_is_eligibility_filter": False,
        "DlyRetx_is_availability_requirement": False,
        "event_reconciliation_in_scope": False,
        "raw_security_keys_may_be_published": False,
        "closed_quality_file_set": list(settings.config["outputs"]["closed_quality_file_set"]),
        "schemas": schemas,
    }
    check_rows = tables["checks"]
    # Checks are the non-overlapping gate layer.  Evidence-table FAIL rows are
    # summarized by their table check, so counting both would double-count one
    # underlying gate failure.
    hard_failures = sum(
        row["status"] == "FAIL" and row["severity"] == "hard"
        for row in check_rows
    )
    fail_count = sum(row["status"] == "FAIL" for row in check_rows)
    pass_count = sum(row["status"] == "PASS" for row in check_rows)
    diagnostic_count = sum(row["status"] == "DIAGNOSTIC" for row in check_rows)
    summary_status = (
        "numeric_unit_primary_passed_pending_independent_verification"
        if mode == "production" and fail_count == 0
        else "numeric_unit_primary_failed"
        if mode == "production"
        else "numeric_unit_discovery_complete_not_authoritative"
    )
    binding = {
        "contract_id": settings.config["upstream"]["contract"]["contract_id"],
        "implementation_sha256": implementation_sha,
        "policy_sha256": policy_sha,
        "empirical_lock_sha256": empirical_lock_sha,
        "profile_config_sha256": profile_sha,
        "snapshot_inventory_sha256": snapshot_inventory_sha,
    }
    downstream_without_id = {
        "schema_version": 1,
        "profile_status": settings.config["status"],
        "typed_market_backbone_authorized": False,
        "universe_unit_inputs_authorized": False,
        "return_history_authorized": False,
        "beta_estimation_authorized": False,
        "labels_authorized": False,
        "backtests_authorized": False,
        "blockers": list(settings.config["downstream"]["blockers"]),
    }
    summary_without_id = {
        "schema_version": 1,
        "mode": mode,
        "status": summary_status,
        "profile_status": settings.config["status"],
        "check_count": len(check_rows),
        "pass_count": pass_count,
        "fail_count": fail_count,
        "diagnostic_count": diagnostic_count,
        "hard_failure_count": hard_failures,
        "aggregate_row_counts": {
            name: len(rows) for name, rows in sorted(tables.items())
        },
        "source_counts": {
            row["source"]: {
                "full_rows": row["full_rows"],
                "retained_rows": row["retained_rows"],
                "canonical_rows": row["canonical_rows"],
            }
            for row in source_receipts
        },
        "production_authorized": mode == "production" and fail_count == 0,
    }
    success_without_id = {
        "schema_version": 1,
        "status": summary_status,
        "closed_quality_file_set": list(settings.config["outputs"]["closed_quality_file_set"]),
        "aggregate_only": True,
        "publication_performed": False,
    }
    # Complete pre-ID semantic seed.  Every deterministic quality/publication
    # semantic is included, while the self-referential build_id is injected
    # only after hashing.  The implementation hash covers renderer logic.
    build_seed = {
        "binding": binding,
        "downstream_gate_status_without_build_id": downstream_without_id,
        "input_receipt": input_receipt,
        "mode": mode,
        "profile_contract": profile_contract,
        "success_without_build_id": success_without_id,
        "summary_without_build_id": summary_without_id,
        "tables": tables,
    }
    build_id = _semantic_build_id(mode, build_seed)
    downstream = {"build_id": build_id, **downstream_without_id}
    summary = {"build_id": build_id, **summary_without_id}
    success = {"build_id": build_id, **success_without_id}
    result = {
        "schema_version": 1,
        "mode": mode,
        "build_id": build_id,
        "binding": binding,
        "summary": summary,
        "profile_contract": profile_contract,
        "input_receipt": input_receipt,
        "downstream_gate_status": downstream,
        "success": success,
        "tables": tables,
    }
    # Rendering is part of the deterministic in-memory result, never publication.
    result["quality_payloads"] = render_quality_payloads(result, settings=settings)
    return result


def run_numeric_unit_discovery(
    settings: NumericUnitSettings | None = None,
) -> dict[str, Any]:
    """Run aggregate-only discovery without writing quality/artifact/pointer state."""
    settings = settings or load_settings()
    if settings.config.get("status") != "discovery_only_unlocked":
        raise NumericUnitProfileError(
            "Discovery requires the explicit discovery_only_unlocked profile status"
        )
    outputs = settings.config["outputs"]
    if any(
        outputs.get(name) is not False
        for name in (
            "discovery_may_write_quality_directory",
            "discovery_may_write_artifact_directory",
            "discovery_may_write_primary_pointer",
            "discovery_may_write_verification_directory",
            "discovery_may_write_verified_pointer",
        )
    ):
        raise NumericUnitProfileError("Discovery write prohibitions are not all false")
    return _run_numeric_unit_scan(settings, mode="discovery")


discover_numeric_unit_profile = run_numeric_unit_discovery


def _require_production_locks(settings: NumericUnitSettings) -> None:
    required_status = settings.config["outputs"]["production_publication_requires_status"]
    if settings.config.get("status") != required_status:
        raise NumericUnitProfileError(
            "Production compilation refused: empirical profile status is unlocked"
        )
    locks = settings.config.get("production_locks", {})
    if set(locks) != PRODUCTION_LOCK_KEYS:
        raise NumericUnitProfileError(
            "Production compilation refused: production lock key set differs"
        )
    if any(value is not True for value in locks.values()):
        raise NumericUnitProfileError(
            "Production compilation refused: every production lock must be true"
        )
    if not settings.authenticate_inputs or not settings.authenticate_upstreams:
        raise NumericUnitProfileError(
            "Production compilation requires authenticated raw inputs and upstreams"
        )
    if settings.config["capitalization_formulas"].get(
        "primary_non_ADR_scale_formula"
    ) != "monthly_interval_formula":
        raise NumericUnitProfileError(
            "Production compilation requires interval ShrOut as the primary "
            "non-ADR capitalization formula"
        )
    _validated_production_expectations(settings)


def compile_numeric_unit_profile(
    settings: NumericUnitSettings | None = None,
) -> dict[str, Any]:
    """Compile a locked primary profile; current discovery config is refused early."""
    settings = settings or load_settings()
    _require_production_locks(settings)
    compiled = _run_numeric_unit_scan(settings, mode="production")
    if (
        compiled["summary"]["fail_count"]
        or compiled["summary"]["hard_failure_count"]
    ):
        raise NumericUnitProfileError(
            "Locked production numeric-unit profile contains failed gates"
        )
    return compiled


def render_quality_payloads(
    compiled: Mapping[str, Any],
    *,
    settings: NumericUnitSettings | None = None,
) -> dict[str, bytes]:
    """Deterministically render the exact closed quality set in memory."""
    settings = settings or load_settings()
    tables = compiled["tables"]
    schemas = settings.config["outputs"]["schemas"]
    payloads = {
        "_SUCCESS.json": canonical_json_bytes(compiled["success"]),
        "summary.json": canonical_json_bytes(compiled["summary"]),
        "profile_contract.json": canonical_json_bytes(compiled["profile_contract"]),
        "input_receipt.json": canonical_json_bytes(compiled["input_receipt"]),
        "downstream_gate_status.json": canonical_json_bytes(compiled["downstream_gate_status"]),
        "checks.csv": _csv_bytes(schemas["checks"], tables["checks"]),
        "numeric_inventory.csv": _csv_bytes(schemas["numeric_inventory"], tables["numeric_inventory"]),
        "return_inventory.csv": _csv_bytes(schemas["return_inventory"], tables["return_inventory"]),
        "identity_inventory.csv": _csv_bytes(schemas["identity_inventory"], tables["identity_inventory"]),
        "capitalization_diagnostics.csv": _csv_bytes(
            schemas["capitalization_diagnostics"], tables["capitalization_diagnostics"]
        ),
    }
    if tuple(sorted(payloads)) != tuple(sorted(QUALITY_FILE_SET)):
        raise NumericUnitProfileError("Rendered quality file set is not closed")
    return dict(sorted(payloads.items()))


def render_artifact_documents(
    compiled: Mapping[str, Any],
    *,
    quality_payloads: Mapping[str, bytes] | None = None,
    settings: NumericUnitSettings | None = None,
) -> dict[str, Any]:
    """Render future publisher inputs without mutating any filesystem path."""
    settings = settings or load_settings()
    quality_payloads = quality_payloads or render_quality_payloads(
        compiled, settings=settings
    )
    contract_id = str(compiled["binding"]["contract_id"])
    quality_directory = _absolute(
        settings.quality_root
        / contract_id
        / settings.config["outputs"]["quality_subdirectory"]
        / str(compiled["build_id"])
    )
    try:
        quality_relative = quality_directory.relative_to(
            _absolute(settings.database_root)
        )
    except ValueError as exc:
        raise NumericUnitProfileError(
            "Rendered quality destination is outside database_root"
        ) from exc
    if not quality_relative.parts or quality_relative.parts[0] != "quality":
        raise NumericUnitProfileError("Rendered quality paths must begin with quality/")
    outputs = [
        {
            "path": (quality_relative / name).as_posix(),
            "sha256": hashlib.sha256(quality_payloads[name]).hexdigest(),
            "size_bytes": len(quality_payloads[name]),
        }
        for name in sorted(quality_payloads)
    ]
    manifest = {
        "binding": compiled["binding"],
        "build_id": compiled["build_id"],
        "contract_id": contract_id,
        "outputs": outputs,
        "profile_name": settings.config["profile_name"],
        "schema_version": 1,
        "status": compiled["summary"]["status"],
        "summary": compiled["summary"],
    }
    manifest_bytes = canonical_json_bytes(manifest)
    inventory = "".join(
        f"{item['sha256']}  {item['path']}\n" for item in outputs
    ).encode("utf-8")
    artifact_payloads = {
        "manifest.json": manifest_bytes,
        "output_hashes.sha256": inventory,
    }
    expected = tuple(sorted(settings.config["outputs"]["closed_artifact_file_set"]))
    if tuple(sorted(artifact_payloads)) != expected:
        raise NumericUnitProfileError("Rendered artifact document set is not closed")
    artifact_directory = _absolute(
        settings.artifact_root
        / settings.config["outputs"]["artifact_subdirectory"]
        / str(compiled["build_id"])
    )
    try:
        artifact_label = artifact_directory.relative_to(
            _absolute(settings.project_root)
        ).as_posix()
    except ValueError as exc:
        raise NumericUnitProfileError(
            "Rendered artifact destination is outside project_root"
        ) from exc
    pointer = {
        "artifact_path": artifact_label,
        "build_id": compiled["build_id"],
        "contract_id": contract_id,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "schema_version": 1,
        "status": compiled["summary"]["status"],
    }
    return {
        "artifact_directory": artifact_directory,
        "artifact_payloads": dict(sorted(artifact_payloads.items())),
        "manifest": manifest,
        "pointer": pointer,
        "pointer_bytes": canonical_json_bytes(pointer),
    }
