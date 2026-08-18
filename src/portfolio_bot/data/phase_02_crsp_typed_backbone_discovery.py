"""Read-only aggregate discovery for CRSP typed-backbone identity joins.

This module exists only to discover and lock the identity-join expectations
needed by the later typed-backbone compiler.  It deliberately does not publish
quality evidence, artifacts, pointers, canonical rows, or raw security keys.

The three raw CSVs are opened once with no-follow semantics.  Their anchor
descriptors remain open for the full run; byte hashing and every CSV pass use
duplicates of those descriptors.  Device, inode, size, modification time, and
change time must agree before open, after open, throughout use, and against the
final pathname.  Out-of-scope payload is CSV-decoded only as an operational
necessity: the scope date(s) are the first and only values inspected before a
batch is filtered.

Daily and monthly grouping never trusts input order.  A first pass writes only
fixed-width typed keys to private temporary partitions and sorts every
partition unconditionally.  A second pass compares all decoded fields for the
sparse duplicate/candidate keys while computing aggregate identity diagnostics
for the selected canonical rows.  Temporary keys are destroyed before the
aggregate JSON result is returned.
"""

from __future__ import annotations

import bisect
import csv
import dataclasses
import datetime as dt
import hashlib
import io
import json
import os
import stat
import tempfile
import tomllib
from collections import Counter, defaultdict
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DEVELOPMENT_CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "phase_02_canonicalization"
    / "crsp_development_v1.toml"
)
DEFAULT_SUPPORTING_CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "phase_02_canonicalization"
    / "crsp_supporting_sources_development_v1.toml"
)
DEFAULT_TYPED_POLICY_CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "phase_02_canonicalization"
    / "crsp_typed_backbone_policy_v1.toml"
)
DEFAULT_TYPED_SCHEMA_CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "phase_02_canonicalization"
    / "crsp_typed_backbone_schema_v1.json"
)
DEFAULT_PATHS_CONFIG = PROJECT_ROOT / "configs" / "paths.toml"
DEFAULT_DATABASE_RELOCATION_RECEIPT = (
    PROJECT_ROOT / "data" / "database_relocation_v1.json"
)
DEFAULT_IDENTITY_EMPIRICAL_LOCK = (
    PROJECT_ROOT
    / "configs"
    / "phase_02_canonicalization"
    / "crsp_typed_backbone_identity_empirical_lock_v1.json"
)

LOCKED_DEVELOPMENT_CONFIG_SHA256 = (
    "645b6ff51480d4e3cbcad313e3488c5a8d45df272ca4af6e7010e870296b4c35"
)
LOCKED_SUPPORTING_CONFIG_SHA256 = (
    "6117495f76106a9a89922c1d2a01d23c824e9e94f783ffd8a8290a99bec11cf1"
)
LOCKED_TYPED_POLICY_CONFIG_SHA256 = (
    "23ec145f587c95b735f675c5ee665d43d506ad2d91d0ae6a46a3942ed5674a89"
)
LOCKED_TYPED_SCHEMA_CONFIG_SHA256 = (
    "62ad603c88798212f60ea4ec9796bffbf65008c6cfd87a08239cddf057e31391"
)
LOCKED_PATHS_CONFIG_SHA256 = (
    "a41135d4b12e3138b65027b5d522e5cb6066ce50e742f878c474ac120a075878"
)
LOCKED_DATABASE_RELOCATION_RECEIPT_SHA256 = (
    "bb4cb92fadfd8df903ec2815f7b11281537e96d542b90c89af8d033321940bec"
)
LOCKED_IDENTITY_EMPIRICAL_LOCK_SHA256 = (
    "f369b9eccf3dbfb9741fcbaa0e0a6a0376d3585c0acb04c2443c94d090a33406"
)
HISTORICAL_PATHS_CONFIG_SHA256 = (
    "f628e2263da59774aff10d480628ba3cebdca555e48765c2507ec6795be9c61c"
)
HISTORICAL_DATABASE_ROOT = "/Users/agraemirolmez/Desktop/database"
CANONICAL_DATABASE_ROOT = "/Users/agraemirolmez/Desktop/Thesis"
PENDING_IDENTITY_COUNT_STATUS = "PENDING_DEVELOPMENT_ONLY_DISCOVERY_NOT_LOCKED"
LOCKED_IDENTITY_COUNT_STATUS = "LOCKED_AFTER_DEVELOPMENT_ONLY_AGGREGATE_DISCOVERY"
FIXTURE_SCOPE_START = dt.date(2000, 1, 1)
FIXTURE_SCOPE_END = dt.date(2014, 12, 31)
FIXTURE_MAX_SOURCE_BYTES = 8 << 20
FIXTURE_MAX_TOTAL_BYTES = 16 << 20
FIXTURE_MAX_ROWS_PER_SOURCE = 100_000
FIXTURE_MAX_TOTAL_FULL_FILE_ROWS = 200_000

SOURCE_ORDER = ("daily", "monthly", "names")
IDENTITY_PAYLOAD_FIELDS = (
    "PERMCO",
    "CUSIP",
    "PrimaryExch",
    "TradingStatusFlg",
    "USIncFlg",
    "IssuerType",
    "SecurityType",
    "SecuritySubType",
    "ShareType",
    "Ticker",
    "SICCD",
    "NAICS",
    "IssuerNm",
)
BYTE_CHUNK_SIZE = 8 << 20
_MIN_ARROW_BLOCK_SIZE = 4096
_EPOCH = dt.date(1970, 1, 1)

_DAILY_DTYPE = np.dtype([("permno", "<i8"), ("calendar", "<i4")], align=False)
_MONTHLY_DTYPE = np.dtype(
    [
        ("permno", "<i8"),
        ("calendar", "<i4"),
        ("interval_start", "<i4"),
        ("interval_end", "<i4"),
    ],
    align=False,
)


class TypedBackboneDiscoveryError(RuntimeError):
    """Raised when discovery cannot produce trustworthy aggregate evidence."""


@dataclasses.dataclass(frozen=True)
class SourceSpec:
    """Authenticated source and exact-schema contract for one raw CSV."""

    name: str
    filename: str
    header: tuple[str, ...]
    sha256: str
    size_bytes: int
    full_file_rows: int | None = None
    retained_rows: int | None = None
    canonical_rows: int | None = None
    duplicate_groups: int | None = None
    duplicate_surplus_rows: int | None = None
    interval_candidate_groups: int | None = None
    candidate_group_size: int | None = None


@dataclasses.dataclass(frozen=True)
class DiscoverySettings:
    """All inputs to the non-publishing scanner.

    ``fixture_mode`` is an explicit API-only test mode.  It still requires
    exact hashes, sizes, headers, stable descriptors, date-first routing, and
    aggregate-only output; it merely permits small caller-supplied schemas and
    omits production row-count locks.
    """

    raw_root: Path
    sources: Mapping[str, SourceSpec]
    start_date: dt.date
    end_date: dt.date
    scratch_root: Path | None = None
    block_size_bytes: int = 16 << 20
    fixture_mode: bool = False
    expected: Mapping[str, Mapping[str, Any]] = dataclasses.field(
        default_factory=dict
    )
    production_config_sha256: Mapping[str, str] = dataclasses.field(
        default_factory=dict
    )
    production_identity_count_status: Mapping[str, str] = dataclasses.field(
        default_factory=dict
    )


def canonical_json_bytes(value: Any) -> bytes:
    """Render the only supported public representation."""

    try:
        text = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise TypedBackboneDiscoveryError(
            f"Discovery result is not canonical JSON: {exc}"
        ) from exc
    return (text + "\n").encode("utf-8")


def _absolute(path: str | Path) -> Path:
    return Path(os.path.abspath(os.fspath(Path(path).expanduser())))


def _read_authenticated_config(
    path: Path,
    *,
    expected_sha256: str,
    format_name: str,
    label: str,
) -> dict[str, Any]:
    """Open, authenticate, and parse one immutable config byte snapshot.

    Hashing and parsing deliberately consume the same in-memory bytes.  The
    pathname is never reopened, and stable descriptor/path metadata is checked
    before and after the single read pass.
    """

    path = _absolute(path)
    _assert_no_symlink_components(path, f"locked {label} config")
    descriptor: int | None = None
    try:
        before = path.lstat()
        if not stat.S_ISREG(before.st_mode):
            raise TypedBackboneDiscoveryError(
                f"Locked {label} config is not a regular file"
            )
        nofollow = getattr(os, "O_NOFOLLOW", None)
        cloexec = getattr(os, "O_CLOEXEC", None)
        if nofollow is None or cloexec is None:
            raise TypedBackboneDiscoveryError(
                "Platform lacks required no-follow config-open semantics"
            )
        descriptor = os.open(path, os.O_RDONLY | nofollow | cloexec)
        opened = os.fstat(descriptor)
        if not _same_file(before, opened):
            raise TypedBackboneDiscoveryError(
                f"Locked {label} config changed while being opened"
            )
        digest = hashlib.sha256()
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, BYTE_CHUNK_SIZE):
            digest.update(chunk)
            chunks.append(chunk)
        payload = b"".join(chunks)
        current = os.fstat(descriptor)
        lexical = path.lstat()
        if not _same_file(before, opened, current, lexical):
            raise TypedBackboneDiscoveryError(
                f"Locked {label} config changed during authentication"
            )
        if len(payload) != int(opened.st_size):
            raise TypedBackboneDiscoveryError(
                f"Locked {label} config byte count differs"
            )
        if digest.hexdigest() != expected_sha256:
            raise TypedBackboneDiscoveryError(
                f"Locked {label} config SHA-256 differs"
            )
    except TypedBackboneDiscoveryError:
        raise
    except OSError as exc:
        raise TypedBackboneDiscoveryError(
            f"Cannot authenticate locked {label} config"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)

    try:
        decoded = payload.decode("utf-8")
        if format_name == "toml":
            value = tomllib.loads(decoded)
        elif format_name == "json":
            value = json.loads(decoded)
        else:
            raise TypedBackboneDiscoveryError("Unknown locked config format")
    except TypedBackboneDiscoveryError:
        raise
    except (UnicodeError, tomllib.TOMLDecodeError, json.JSONDecodeError) as exc:
        raise TypedBackboneDiscoveryError(
            f"Cannot parse authenticated locked {label} config"
        ) from exc
    if not isinstance(value, dict):
        raise TypedBackboneDiscoveryError(
            f"Authenticated locked {label} config has the wrong root type"
        )
    return value


def _locked_production_config_sha256() -> dict[str, str]:
    return {
        "development_profile": LOCKED_DEVELOPMENT_CONFIG_SHA256,
        "supporting_profile": LOCKED_SUPPORTING_CONFIG_SHA256,
        "typed_policy": LOCKED_TYPED_POLICY_CONFIG_SHA256,
        "typed_schema": LOCKED_TYPED_SCHEMA_CONFIG_SHA256,
        "paths": LOCKED_PATHS_CONFIG_SHA256,
        "database_relocation": LOCKED_DATABASE_RELOCATION_RECEIPT_SHA256,
        "identity_empirical_lock": LOCKED_IDENTITY_EMPIRICAL_LOCK_SHA256,
    }


def _assert_database_relocation(
    *,
    policy: Mapping[str, Any],
    relocation: Mapping[str, Any],
    paths: Mapping[str, Any],
) -> None:
    """Authenticate the one permitted rename without rewriting frozen evidence."""

    try:
        policy_relocation = policy["upstream"]["database_relocation"]
        policy_phase_01 = policy["upstream"]["phase_01"]
        relocation_paths = relocation["paths_configuration"]
        relocation_snapshot = relocation["frozen_snapshot"]
        relocation_evidence = relocation["relocation_evidence"]
        historical_policy = relocation["historical_evidence_policy"]
        configured_root = str(paths["database"]["root"])
    except (KeyError, TypeError) as exc:
        raise TypedBackboneDiscoveryError(
            "Database relocation contract is structurally incomplete"
        ) from exc

    exact_pairs = (
        (policy_relocation.get("relocation_id"), relocation.get("relocation_id")),
        (policy_relocation.get("required_status"), relocation.get("status")),
        (
            policy_relocation.get("receipt_sha256"),
            LOCKED_DATABASE_RELOCATION_RECEIPT_SHA256,
        ),
        (
            policy_relocation.get("historical_paths_config_sha256"),
            HISTORICAL_PATHS_CONFIG_SHA256,
        ),
        (
            policy_relocation.get("current_paths_config_sha256"),
            LOCKED_PATHS_CONFIG_SHA256,
        ),
        (policy_relocation.get("old_database_root"), HISTORICAL_DATABASE_ROOT),
        (policy_relocation.get("new_database_root"), CANONICAL_DATABASE_ROOT),
        (relocation.get("old_database_root"), HISTORICAL_DATABASE_ROOT),
        (relocation.get("new_database_root"), CANONICAL_DATABASE_ROOT),
        (
            relocation_paths.get("historical_frozen_sha256"),
            HISTORICAL_PATHS_CONFIG_SHA256,
        ),
        (relocation_paths.get("current_sha256"), LOCKED_PATHS_CONFIG_SHA256),
        (relocation_snapshot.get("raw_snapshot_id"), policy_phase_01.get("raw_snapshot_id")),
        (relocation_snapshot.get("freeze_id"), policy_phase_01.get("freeze_id")),
        (
            relocation_snapshot.get("freeze_manifest_sha256"),
            policy_phase_01.get("freeze_manifest_sha256"),
        ),
        (configured_root, CANONICAL_DATABASE_ROOT),
    )
    if any(actual != expected for actual, expected in exact_pairs):
        raise TypedBackboneDiscoveryError("Database relocation binding differs")
    if (
        relocation.get("old_root_exists_at_recording") is not False
        or relocation.get("new_root_exists_at_recording") is not True
        or relocation_evidence.get("raw_registry_matches_frozen_snapshot") is not True
        or relocation_evidence.get("current_csv_file_count") != 25
        or relocation_evidence.get("all_frozen_stat_identities_match") is not True
        or relocation_evidence.get("core_hashes_match_frozen_contracts") is not True
        or historical_policy.get("rewrite_sealed_artifacts") is not False
        or historical_policy.get("new_runs_must_use_new_database_root") is not True
        or historical_policy.get(
            "relocation_receipt_required_for_new_runs_that_bind_pre_relocation_evidence"
        )
        is not True
    ):
        raise TypedBackboneDiscoveryError("Database relocation evidence differs")


def _assert_typed_stage_contract(
    *,
    policy: Mapping[str, Any],
    schema: Mapping[str, Any],
    development: Mapping[str, Any],
    supporting: Mapping[str, Any],
    identity_lock: Mapping[str, Any],
    relocation: Mapping[str, Any],
    paths: Mapping[str, Any],
) -> dict[str, str]:
    """Authenticate the typed stage and its now-frozen identity expectations."""

    try:
        policy_schema = policy["schema"]
        upstream_phase_02 = policy["upstream"]["phase_02"]
        upstream_daily_monthly = policy["upstream"]["daily_monthly_profile"]
        upstream_supporting = policy["upstream"]["supporting_profile"]
        upstream_identity_lock = policy["upstream"]["identity_empirical_lock"]
        policy_scope = policy["scope"]
        schema_scope = schema["scope"]
        development_scope = development["scope"]
        daily_source = development["sources"]["daily"]
        monthly_source = development["sources"]["monthly"]
        supporting_sources = {
            str(value["filename"]): value for value in supporting["sources"]
        }
        names_source = supporting_sources["crsp_names_ciz.csv"]
        source_access = policy["source_access"]
        joins = policy["joins"]
        policy_monthly_canonicalization = policy["canonicalization"]["monthly"]
        policy_identity_canonicalization = policy["canonicalization"]["identity"]
        development_monthly_candidates = development["expected"][
            "monthly_canonical_key_after_raw_exact_collapse"
        ]
        locked_expectations = identity_lock["aggregate_expectations"]
    except (KeyError, TypeError, ValueError) as exc:
        raise TypedBackboneDiscoveryError(
            "Typed stage contract is structurally incomplete"
        ) from exc

    required_status = "locked_preproduction_before_typed_backbone_build"
    _assert_database_relocation(policy=policy, relocation=relocation, paths=paths)
    if (
        policy.get("status") != required_status
        or policy_schema.get("required_status") != required_status
        or schema.get("status") != required_status
        or policy_schema.get("schema_id") != schema.get("schema_id")
    ):
        raise TypedBackboneDiscoveryError("Typed policy/schema stage status differs")
    if (
        policy_schema.get("sha256") != LOCKED_TYPED_SCHEMA_CONFIG_SHA256
        or upstream_phase_02.get("paths_config_sha256")
        != LOCKED_PATHS_CONFIG_SHA256
        or upstream_daily_monthly.get("config_sha256")
        != LOCKED_DEVELOPMENT_CONFIG_SHA256
        or upstream_supporting.get("config_sha256")
        != LOCKED_SUPPORTING_CONFIG_SHA256
        or upstream_identity_lock.get("sha256")
        != LOCKED_IDENTITY_EMPIRICAL_LOCK_SHA256
        or upstream_identity_lock.get("lock_id") != identity_lock.get("lock_id")
        or upstream_identity_lock.get("required_status")
        != identity_lock.get("status")
    ):
        raise TypedBackboneDiscoveryError("Typed policy config binding differs")

    expected_start = str(development_scope["start_date"])
    expected_end = str(development_scope["end_date"])
    if (
        str(policy_scope.get("start_date")) != expected_start
        or str(policy_scope.get("quality_end_date")) != expected_end
        or str(schema_scope.get("start_date")) != expected_start
        or str(schema_scope.get("quality_end_date")) != expected_end
    ):
        raise TypedBackboneDiscoveryError("Typed policy/schema scope differs")

    source_pairs = (
        ("daily", source_access["daily"], daily_source),
        ("monthly", source_access["monthly"], monthly_source),
        ("identity", source_access["identity"], names_source),
    )
    for label, typed_source, profile_source in source_pairs:
        try:
            agrees = all(
                typed_source[field] == profile_source[field]
                for field in ("filename", "sha256", "size_bytes", "full_file_rows")
            )
        except (KeyError, TypeError) as exc:
            raise TypedBackboneDiscoveryError(
                f"Typed {label} source binding is incomplete"
            ) from exc
        if not agrees:
            raise TypedBackboneDiscoveryError(
                f"Typed {label} source binding differs from authenticated profile"
            )

    for field in (
        "effective_interval_candidate_groups",
        "candidate_group_size",
        "output_rows",
    ):
        if policy_monthly_canonicalization.get(field) != development_monthly_candidates.get(
            field
        ):
            raise TypedBackboneDiscoveryError(
                f"Typed monthly candidate contract differs for {field}"
            )
    if development_monthly_candidates.get("candidate_group_size") != 2:
        raise TypedBackboneDiscoveryError(
            "Locked monthly candidate group size is not exactly two"
        )

    statuses: dict[str, str] = {}
    count_fields = {
        "daily": (
            "population_rows",
            "blank_as_of_date_rows",
            "one_match_rows",
            "zero_match_rows",
            "multiple_match_rows",
            "embedded_interval_exact_key_match_rows",
            "embedded_interval_exact_key_missing_rows",
            "as_of_interval_key_equal_rows",
            "as_of_interval_key_different_rows",
            "as_of_payload_all_fields_equal_rows",
            "as_of_payload_one_or_more_fields_different_rows",
        ),
        "monthly": (
            "population_rows",
            "blank_as_of_date_rows",
            "nonblank_as_of_date_rows",
            "one_match_rows",
            "zero_match_rows",
            "zero_match_rows_with_nonblank_as_of_date",
            "multiple_match_rows",
            "embedded_interval_exact_key_match_rows",
            "embedded_interval_exact_key_missing_rows",
            "calendar_month_substitution_for_blank_as_of_date_permitted",
            "as_of_interval_key_equal_rows",
            "as_of_interval_key_different_rows",
            "as_of_payload_all_fields_equal_rows",
            "as_of_payload_one_or_more_fields_different_rows",
        ),
    }
    for frequency, join_name in (
        ("daily", "daily_to_identity"),
        ("monthly", "monthly_to_identity"),
    ):
        try:
            join = joins[join_name]
            status_value = str(join["expected_match_counts_status"])
            empirical = locked_expectations[join_name]
        except (KeyError, TypeError) as exc:
            raise TypedBackboneDiscoveryError(
                f"Typed {frequency} identity-count stage is incomplete"
            ) from exc
        if (
            status_value != LOCKED_IDENTITY_COUNT_STATUS
            or join.get("production_publication_empirical_expectation_is_locked")
            is not True
            or any(join.get(field) != empirical.get(field) for field in count_fields[frequency])
        ):
            raise TypedBackboneDiscoveryError(
                f"Typed {frequency} identity counts differ from the empirical lock"
            )
        statuses[frequency] = status_value

    identity_intervals = locked_expectations["identity_intervals"]
    policy_identity_fields = {
        "retained_rows": "output_rows",
        "unique_permnos": "unique_permnos",
        "duplicate_key_groups": "duplicate_key_groups",
        "contiguous_edges": "contiguous_edges",
        "gap_edges": "gap_edges",
        "overlap_edges": "overlap_edges",
    }
    if any(
        identity_intervals.get(lock_field)
        != policy_identity_canonicalization.get(policy_field)
        for lock_field, policy_field in policy_identity_fields.items()
    ):
        raise TypedBackboneDiscoveryError(
            "Typed identity-interval counts differ from the empirical lock"
        )
    return statuses


def _source_from_config(
    name: str,
    source: Mapping[str, Any],
    *,
    expected: Mapping[str, int],
) -> SourceSpec:
    return SourceSpec(
        name=name,
        filename=str(source["filename"]),
        header=tuple(str(value) for value in source["header"]),
        sha256=str(source["sha256"]),
        size_bytes=int(source["size_bytes"]),
        full_file_rows=int(source["full_file_rows"]),
        retained_rows=int(expected["retained_rows"]),
        canonical_rows=int(expected.get("canonical_rows", expected["retained_rows"])),
        duplicate_groups=int(expected.get("duplicate_groups", 0)),
        duplicate_surplus_rows=int(expected.get("duplicate_surplus_rows", 0)),
        interval_candidate_groups=int(expected.get("interval_candidate_groups", 0)),
        candidate_group_size=(
            int(expected["candidate_group_size"])
            if "candidate_group_size" in expected
            else None
        ),
    )


def load_production_settings(
    *,
    development_config_path: Path = DEFAULT_DEVELOPMENT_CONFIG,
    supporting_config_path: Path = DEFAULT_SUPPORTING_CONFIG,
    typed_policy_config_path: Path = DEFAULT_TYPED_POLICY_CONFIG,
    typed_schema_config_path: Path = DEFAULT_TYPED_SCHEMA_CONFIG,
    paths_config_path: Path = DEFAULT_PATHS_CONFIG,
    database_relocation_path: Path = DEFAULT_DATABASE_RELOCATION_RECEIPT,
    identity_empirical_lock_path: Path = DEFAULT_IDENTITY_EMPIRICAL_LOCK,
    scratch_root: Path | None = None,
    block_size_bytes: int = 16 << 20,
) -> DiscoverySettings:
    """Load the exact checked-in locks for an authenticated production scan."""

    development_config_path = _absolute(development_config_path)
    supporting_config_path = _absolute(supporting_config_path)
    typed_policy_config_path = _absolute(typed_policy_config_path)
    typed_schema_config_path = _absolute(typed_schema_config_path)
    paths_config_path = _absolute(paths_config_path)
    database_relocation_path = _absolute(database_relocation_path)
    identity_empirical_lock_path = _absolute(identity_empirical_lock_path)
    development = _read_authenticated_config(
        development_config_path,
        expected_sha256=LOCKED_DEVELOPMENT_CONFIG_SHA256,
        format_name="toml",
        label="development",
    )
    supporting = _read_authenticated_config(
        supporting_config_path,
        expected_sha256=LOCKED_SUPPORTING_CONFIG_SHA256,
        format_name="toml",
        label="supporting",
    )
    typed_policy = _read_authenticated_config(
        typed_policy_config_path,
        expected_sha256=LOCKED_TYPED_POLICY_CONFIG_SHA256,
        format_name="toml",
        label="typed policy",
    )
    typed_schema = _read_authenticated_config(
        typed_schema_config_path,
        expected_sha256=LOCKED_TYPED_SCHEMA_CONFIG_SHA256,
        format_name="json",
        label="typed schema",
    )
    paths = _read_authenticated_config(
        paths_config_path,
        expected_sha256=LOCKED_PATHS_CONFIG_SHA256,
        format_name="toml",
        label="paths",
    )
    database_relocation = _read_authenticated_config(
        database_relocation_path,
        expected_sha256=LOCKED_DATABASE_RELOCATION_RECEIPT_SHA256,
        format_name="json",
        label="database relocation",
    )
    identity_empirical_lock = _read_authenticated_config(
        identity_empirical_lock_path,
        expected_sha256=LOCKED_IDENTITY_EMPIRICAL_LOCK_SHA256,
        format_name="json",
        label="identity empirical lock",
    )
    identity_count_status = _assert_typed_stage_contract(
        policy=typed_policy,
        schema=typed_schema,
        development=development,
        supporting=supporting,
        identity_lock=identity_empirical_lock,
        relocation=database_relocation,
        paths=paths,
    )

    daily_source = development["sources"]["daily"]
    monthly_source = development["sources"]["monthly"]
    supporting_sources = {
        str(value["filename"]): value for value in supporting["sources"]
    }
    names_source = supporting_sources.get("crsp_names_ciz.csv")
    if names_source is None:
        raise TypedBackboneDiscoveryError("Locked names source is absent")

    expected = {
        "daily": {
            "full_file_rows": int(daily_source["full_file_rows"]),
            "retained_rows": int(development["expected"]["daily_raw_key"]["input_rows"]),
            "canonical_rows": int(development["expected"]["daily_raw_key"]["output_rows"]),
            "duplicate_groups": int(development["expected"]["daily_raw_key"]["duplicate_groups"]),
            "duplicate_surplus_rows": int(development["expected"]["daily_raw_key"]["surplus_rows"]),
            "conflicting_groups": int(development["expected"]["daily_raw_key"]["conflicting_groups"]),
        },
        "monthly": {
            "full_file_rows": int(monthly_source["full_file_rows"]),
            "retained_rows": int(development["expected"]["monthly_raw_key"]["input_rows"]),
            "raw_distinct_rows": int(development["expected"]["monthly_raw_key"]["output_rows"]),
            "canonical_rows": int(development["expected"]["monthly_canonical_key_after_raw_exact_collapse"]["output_rows"]),
            "duplicate_groups": int(development["expected"]["monthly_raw_key"]["duplicate_groups"]),
            "duplicate_surplus_rows": int(development["expected"]["monthly_raw_key"]["surplus_rows"]),
            "conflicting_groups": int(development["expected"]["monthly_raw_key"]["conflicting_groups"]),
            "interval_candidate_groups": int(development["expected"]["monthly_canonical_key_after_raw_exact_collapse"]["effective_interval_candidate_groups"]),
            "candidate_group_size": int(development["expected"]["monthly_canonical_key_after_raw_exact_collapse"]["candidate_group_size"]),
            "candidate_size_histogram": {
                str(development["expected"]["monthly_canonical_key_after_raw_exact_collapse"]["candidate_group_size"]): int(
                    development["expected"]["monthly_canonical_key_after_raw_exact_collapse"]["effective_interval_candidate_groups"]
                )
            },
            "groups_with_exactly_one_price_date_match": int(development["expected"]["monthly_canonical_key_after_raw_exact_collapse"]["groups_with_exactly_one_price_date_match"]),
            "groups_with_zero_price_date_matches": int(development["expected"]["monthly_canonical_key_after_raw_exact_collapse"]["groups_with_zero_price_date_matches"]),
            "groups_with_multiple_price_date_matches": int(development["expected"]["monthly_canonical_key_after_raw_exact_collapse"]["groups_with_multiple_price_date_matches"]),
            "groups_with_exactly_one_calendar_date_match": int(development["expected"]["monthly_canonical_key_after_raw_exact_collapse"]["groups_with_exactly_one_calendar_date_match"]),
            "groups_with_zero_calendar_date_matches": int(development["expected"]["monthly_canonical_key_after_raw_exact_collapse"]["groups_with_zero_calendar_date_matches"]),
            "groups_with_multiple_calendar_date_matches": int(development["expected"]["monthly_canonical_key_after_raw_exact_collapse"]["groups_with_multiple_calendar_date_matches"]),
        },
        "names": {
            "full_file_rows": int(names_source["full_file_rows"]),
            "retained_rows": int(supporting["expected"]["crsp_names_ciz_csv"]["retained_rows"]),
            "unique_permnos": int(supporting["expected"]["crsp_names_ciz_csv"]["unique_permnos"]),
            "contiguous_edges": int(supporting["expected"]["crsp_names_ciz_csv"]["contiguous_edges"]),
            "gap_edges": int(supporting["expected"]["crsp_names_ciz_csv"]["gap_edges"]),
            "overlap_edges": int(supporting["expected"]["crsp_names_ciz_csv"]["overlap_edges"]),
        },
    }

    database_root = _absolute(
        os.environ.get("PORTFOLIO_DATABASE_ROOT", paths["database"]["root"])
    )
    if database_root == _absolute(HISTORICAL_DATABASE_ROOT):
        raise TypedBackboneDiscoveryError(
            "The historical Desktop/database root is no longer permitted"
        )
    raw_root = database_root / str(paths["database"]["raw"])
    sources = {
        "daily": _source_from_config(
            "daily",
            daily_source,
            expected={
                "retained_rows": expected["daily"]["retained_rows"],
                "canonical_rows": expected["daily"]["canonical_rows"],
                "duplicate_groups": expected["daily"]["duplicate_groups"],
                "duplicate_surplus_rows": expected["daily"]["duplicate_surplus_rows"],
            },
        ),
        "monthly": _source_from_config(
            "monthly",
            monthly_source,
            expected={
                "retained_rows": expected["monthly"]["retained_rows"],
                "canonical_rows": expected["monthly"]["canonical_rows"],
                "duplicate_groups": expected["monthly"]["duplicate_groups"],
                "duplicate_surplus_rows": expected["monthly"]["duplicate_surplus_rows"],
                "interval_candidate_groups": expected["monthly"]["interval_candidate_groups"],
                "candidate_group_size": expected["monthly"]["candidate_group_size"],
            },
        ),
        "names": _source_from_config(
            "names",
            names_source,
            expected={"retained_rows": expected["names"]["retained_rows"]},
        ),
    }
    scope = development["scope"]
    return DiscoverySettings(
        raw_root=raw_root,
        sources=sources,
        start_date=dt.date.fromisoformat(str(scope["start_date"])),
        end_date=dt.date.fromisoformat(str(scope["end_date"])),
        scratch_root=_absolute(scratch_root) if scratch_root is not None else None,
        block_size_bytes=block_size_bytes,
        fixture_mode=False,
        expected=expected,
        production_config_sha256=_locked_production_config_sha256(),
        production_identity_count_status=identity_count_status,
    )


def make_fixture_settings(
    *,
    raw_root: Path,
    sources: Mapping[str, SourceSpec],
    start_date: dt.date,
    end_date: dt.date,
    scratch_root: Path | None = None,
    block_size_bytes: int = 4096,
) -> DiscoverySettings:
    """Construct explicit small-fixture settings without weakening authentication."""

    if type(sources) is not dict:
        raise TypedBackboneDiscoveryError(
            "Fixture sources must be supplied as an exact dict"
        )
    settings = DiscoverySettings(
        raw_root=_absolute(raw_root),
        sources=dict(sources),
        start_date=start_date,
        end_date=end_date,
        scratch_root=_absolute(scratch_root) if scratch_root is not None else None,
        block_size_bytes=block_size_bytes,
        fixture_mode=True,
        expected={},
    )
    _assert_fixture_settings(settings)
    return settings


def _same_file(*values: os.stat_result) -> bool:
    if not values or any(not stat.S_ISREG(value.st_mode) for value in values):
        return False
    first = values[0]
    fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    return all(
        all(getattr(value, field) == getattr(first, field) for field in fields)
        for value in values[1:]
    )


def _assert_no_symlink_components(path: Path, label: str) -> None:
    absolute = _absolute(path)
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current = current / component
        try:
            metadata = current.lstat()
        except OSError as exc:
            raise TypedBackboneDiscoveryError(
                f"Cannot authenticate {label} path components"
            ) from exc
        if stat.S_ISLNK(metadata.st_mode):
            raise TypedBackboneDiscoveryError(f"Symlink is prohibited in {label} path")


@dataclasses.dataclass
class _AuthenticatedSource:
    path: Path
    spec: SourceSpec
    descriptor: int
    before: os.stat_result
    opened: os.stat_result
    sha256: str
    closed: bool = False

    @classmethod
    def open(cls, raw_root: Path, spec: SourceSpec) -> "_AuthenticatedSource":
        if (
            Path(spec.filename).name != spec.filename
            or "/" in spec.filename
            or "\\" in spec.filename
            or spec.filename in {".", ".."}
        ):
            raise TypedBackboneDiscoveryError("Raw filename is not a direct child name")
        root = _absolute(raw_root)
        path = _absolute(root / spec.filename)
        if path.parent != root:
            raise TypedBackboneDiscoveryError("Raw source escapes raw_root")
        _assert_no_symlink_components(path, "raw source")
        try:
            before = path.lstat()
            if not stat.S_ISREG(before.st_mode):
                raise TypedBackboneDiscoveryError("Raw source is not a regular file")
            flags = (
                os.O_RDONLY
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0)
            )
            descriptor = os.open(path, flags)
            try:
                opened = os.fstat(descriptor)
                if not _same_file(before, opened):
                    raise TypedBackboneDiscoveryError(
                        "Raw source changed while being opened"
                    )
                digest = hashlib.sha256()
                os.lseek(descriptor, 0, os.SEEK_SET)
                while chunk := os.read(descriptor, BYTE_CHUNK_SIZE):
                    digest.update(chunk)
                sha256 = digest.hexdigest()
                current = os.fstat(descriptor)
                if not _same_file(before, opened, current, path.lstat()):
                    raise TypedBackboneDiscoveryError(
                        "Raw source changed during authentication"
                    )
                if int(opened.st_size) != spec.size_bytes:
                    raise TypedBackboneDiscoveryError(
                        f"Authenticated size differs for {spec.filename}"
                    )
                if sha256 != spec.sha256:
                    raise TypedBackboneDiscoveryError(
                        f"Authenticated SHA-256 differs for {spec.filename}"
                    )
                result = cls(path, spec, descriptor, before, opened, sha256)
                result._check_header()
                return result
            except BaseException:
                os.close(descriptor)
                raise
        except TypedBackboneDiscoveryError:
            raise
        except OSError as exc:
            raise TypedBackboneDiscoveryError(
                f"Cannot open authenticated raw source {spec.filename}"
            ) from exc

    def __enter__(self) -> "_AuthenticatedSource":
        return self

    def __exit__(self, *_: Any) -> bool:
        self.close()
        return False

    def assert_stable(self) -> None:
        if self.closed:
            raise TypedBackboneDiscoveryError("Raw anchor descriptor is closed")
        _assert_no_symlink_components(self.path, "raw source")
        try:
            current = os.fstat(self.descriptor)
            lexical = self.path.lstat()
        except OSError as exc:
            raise TypedBackboneDiscoveryError("Cannot restat authenticated raw source") from exc
        if not _same_file(self.before, self.opened, current, lexical):
            raise TypedBackboneDiscoveryError(
                f"Authenticated raw source changed during discovery: {self.spec.filename}"
            )

    def close(self) -> None:
        if self.closed:
            return
        failure: BaseException | None = None
        try:
            self.assert_stable()
        except BaseException as exc:
            failure = exc
        try:
            os.close(self.descriptor)
        finally:
            self.closed = True
        if failure is not None:
            raise failure

    def _duplicate_handle(self) -> io.BufferedReader:
        self.assert_stable()
        descriptor = os.dup(self.descriptor)
        try:
            os.lseek(descriptor, 0, os.SEEK_SET)
            return os.fdopen(descriptor, "rb", buffering=0)
        except BaseException:
            os.close(descriptor)
            raise

    def _check_header(self) -> None:
        handle = self._duplicate_handle()
        try:
            line = handle.readline(1 << 20)
            if len(line) == 1 << 20 and not line.endswith((b"\n", b"\r")):
                raise TypedBackboneDiscoveryError("CSV header exceeds bounded limit")
            actual = next(csv.reader([line.decode("utf-8")]))
        except TypedBackboneDiscoveryError:
            raise
        except (OSError, UnicodeError, csv.Error, StopIteration) as exc:
            raise TypedBackboneDiscoveryError(
                f"Cannot decode exact header for {self.spec.filename}"
            ) from exc
        finally:
            handle.close()
        if actual != list(self.spec.header):
            raise TypedBackboneDiscoveryError(
                f"Exact CSV header differs for {self.spec.filename}"
            )

    @contextmanager
    def batches(self, selected: Sequence[str], block_size_bytes: int) -> Iterator[Iterable[pa.RecordBatch]]:
        selected = tuple(dict.fromkeys(selected))
        if any(column not in self.spec.header for column in selected):
            raise TypedBackboneDiscoveryError(
                f"Requested discovery column is absent from {self.spec.filename}"
            )
        handle = self._duplicate_handle()
        try:
            try:
                reader = pacsv.open_csv(
                    handle,
                    read_options=pacsv.ReadOptions(
                        block_size=max(block_size_bytes, _MIN_ARROW_BLOCK_SIZE),
                        use_threads=False,
                        encoding="utf8",
                    ),
                    parse_options=pacsv.ParseOptions(
                        delimiter=",",
                        quote_char='"',
                        double_quote=True,
                        escape_char=False,
                        newlines_in_values=True,
                        ignore_empty_lines=False,
                    ),
                    convert_options=pacsv.ConvertOptions(
                        include_columns=list(selected),
                        column_types={column: pa.string() for column in selected},
                        strings_can_be_null=False,
                        quoted_strings_can_be_null=False,
                        null_values=[],
                        # Payload is decoded only after date-first filtering.
                        # This permits operational CSV tokenization of excluded
                        # bytes without semantically inspecting or aggregating
                        # an out-of-scope non-date field.
                        check_utf8=False,
                    ),
                )
            except (OSError, pa.ArrowException):
                raise TypedBackboneDiscoveryError(
                    f"Cannot open aggregate CSV stream for {self.spec.filename}"
                ) from None

            def iterator() -> Iterator[pa.RecordBatch]:
                while True:
                    try:
                        yield reader.read_next_batch()
                    except StopIteration:
                        return
                    except (OSError, UnicodeError, pa.ArrowException):
                        raise TypedBackboneDiscoveryError(
                            f"Aggregate CSV stream failed for {self.spec.filename}"
                        ) from None

            yield iterator()
        finally:
            handle.close()
            self.assert_stable()


def _strict_date32(values: pa.Array, label: str) -> pa.Array:
    lexical = pc.match_substring_regex(values, r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$")
    parsed = pc.strptime(values, format="%Y-%m-%d", unit="s", error_is_null=True)
    round_trip = pc.strftime(parsed, format="%Y-%m-%d")
    valid = pc.and_(
        pc.and_(lexical, pc.is_valid(parsed)),
        pc.fill_null(pc.equal(round_trip, values), False),
    )
    mask = np.asarray(valid.to_numpy(zero_copy_only=False), dtype=bool)
    if not bool(mask.all()):
        blank = int(pc.sum(pc.cast(pc.equal(values, ""), pa.int64())).as_py() or 0)
        invalid = len(values) - int(mask.sum()) - blank
        raise TypedBackboneDiscoveryError(
            f"{label} contains blank or invalid dates (blank={blank},invalid={invalid})"
        )
    return pc.cast(parsed, pa.date32())


def _positive_permnos(values: pa.Array, label: str) -> np.ndarray:
    lexical = np.asarray(
        pc.match_substring_regex(values, r"^[0-9]+$").to_numpy(zero_copy_only=False),
        dtype=bool,
    )
    if not bool(lexical.all()):
        raise TypedBackboneDiscoveryError(f"{label} contains invalid PERMNO values")
    try:
        parsed = pc.cast(values, pa.int64(), safe=True)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
        raise TypedBackboneDiscoveryError(f"{label} contains PERMNO outside int64") from None
    if bool(np.asarray(pc.less_equal(parsed, 0).to_numpy(zero_copy_only=False)).any()):
        raise TypedBackboneDiscoveryError(f"{label} contains nonpositive PERMNO")
    return np.asarray(parsed.to_numpy(zero_copy_only=False), dtype=np.int64)


def _date_days(values: pa.Array) -> np.ndarray:
    return (
        np.asarray(values.to_numpy(zero_copy_only=False))
        .astype("datetime64[D]")
        .astype(np.int32)
    )


def _filter(batch: pa.RecordBatch, mask: pa.Array) -> pa.RecordBatch:
    return batch.filter(mask) if bool(pc.any(mask).as_py()) else batch.slice(0, 0)


def _decoded_retained_rows(
    retained: pa.RecordBatch,
    label: str,
) -> Iterator[tuple[str, ...]]:
    try:
        columns = [column.to_pylist() for column in retained.columns]
    except (UnicodeError, pa.ArrowException):
        raise TypedBackboneDiscoveryError(
            f"Retained {label} payload is not valid decoded UTF-8"
        ) from None
    return zip(*columns, strict=True)


def _scope_by_date(
    batch: pa.RecordBatch,
    date_column: str,
    start: dt.date,
    end: dt.date,
    label: str,
) -> tuple[pa.RecordBatch, pa.Array]:
    # No non-date column may be inspected before this filter.
    raw = batch.column(batch.schema.get_field_index(date_column))
    parsed = _strict_date32(raw, f"{label} scope date")
    mask = pc.and_(pc.greater_equal(raw, start.isoformat()), pc.less_equal(raw, end.isoformat()))
    return _filter(batch, mask), pc.filter(parsed, mask)


def _optional_date_days(values: pa.Array, label: str) -> list[int | None]:
    raw = values.to_pylist()
    result: list[int | None] = []
    nonblank_values = [value for value in raw if value != ""]
    if nonblank_values:
        _strict_date32(pa.array(nonblank_values, type=pa.string()), label)
    for value in raw:
        result.append(None if value == "" else (dt.date.fromisoformat(value) - _EPOCH).days)
    return result


class _PackedPartitions:
    """Private sorted-key scratch whose paths and records never escape."""

    def __init__(self, root: Path, dtype: np.dtype) -> None:
        self.root = root
        self.dtype = dtype
        root.mkdir(mode=0o700, parents=True, exist_ok=False)
        self.paths: dict[int, Path] = {}

    def append(self, records: np.ndarray) -> None:
        if records.dtype != self.dtype:
            raise TypedBackboneDiscoveryError("Packed key dtype differs")
        if len(records) == 0:
            return
        partitions = np.right_shift(records["permno"], 8)
        order = np.argsort(partitions, kind="stable")
        ordered = partitions[order]
        starts = np.concatenate(
            (np.array([0], dtype=np.int64), np.flatnonzero(ordered[1:] != ordered[:-1]) + 1)
        )
        ends = np.concatenate((starts[1:], np.array([len(order)], dtype=np.int64)))
        for start, end in zip(starts.tolist(), ends.tolist(), strict=True):
            partition = int(ordered[start])
            path = self.paths.setdefault(
                partition, self.root / f"p-{partition:020d}.bin"
            )
            try:
                with path.open("ab") as handle:
                    handle.write(np.ascontiguousarray(records[order[start:end]]).tobytes())
                path.chmod(0o600)
            except OSError as exc:
                raise TypedBackboneDiscoveryError("Cannot write private key scratch") from exc

    def sorted_records(self) -> Iterator[np.ndarray]:
        fields = list(self.dtype.names or ())
        for partition in sorted(self.paths):
            path = self.paths[partition]
            try:
                if path.stat().st_size % self.dtype.itemsize:
                    raise TypedBackboneDiscoveryError("Private key scratch is truncated")
                records = np.fromfile(path, dtype=self.dtype).copy()
            except OSError as exc:
                raise TypedBackboneDiscoveryError("Cannot read private key scratch") from exc
            records.sort(order=fields, kind="stable")
            yield records


def _group_starts(records: np.ndarray, fields: Sequence[str]) -> np.ndarray:
    if len(records) == 0:
        return np.empty(0, dtype=np.int64)
    same = np.ones(len(records) - 1, dtype=bool)
    for field in fields:
        same &= records[field][1:] == records[field][:-1]
    return np.concatenate((np.array([0], dtype=np.int64), np.flatnonzero(~same) + 1))


def _summarize_partitions(
    partitions: _PackedPartitions,
    fields: Sequence[str],
    *,
    collapse_raw_monthly_first: bool = False,
) -> tuple[dict[str, Any], dict[tuple[int, ...], int]]:
    histogram: Counter[int] = Counter()
    distinct = duplicate_groups = affected = surplus = maximum = 0
    duplicate_keys: dict[tuple[int, ...], int] = {}
    for raw_records in partitions.sorted_records():
        records = raw_records
        if collapse_raw_monthly_first:
            raw_starts = _group_starts(
                records, ("permno", "calendar", "interval_start", "interval_end")
            )
            records = records[raw_starts]
        starts = _group_starts(records, fields)
        if len(starts) == 0:
            continue
        ends = np.concatenate((starts[1:], np.array([len(records)], dtype=np.int64)))
        sizes = ends - starts
        keys = records[starts]
        unique_sizes, counts = np.unique(sizes, return_counts=True)
        histogram.update(
            {int(size): int(count) for size, count in zip(unique_sizes, counts, strict=True)}
        )
        distinct += len(sizes)
        maximum = max(maximum, int(sizes.max()))
        duplicate = sizes > 1
        duplicate_sizes = sizes[duplicate]
        duplicate_groups += int(duplicate.sum())
        affected += int(duplicate_sizes.sum())
        surplus += int((duplicate_sizes - 1).sum())
        for key, size in zip(keys[duplicate], duplicate_sizes, strict=True):
            typed = tuple(int(key[field]) for field in fields)
            if typed in duplicate_keys:
                raise TypedBackboneDiscoveryError("Private duplicate key spans partitions")
            duplicate_keys[typed] = int(size)
    return (
        {
            "distinct_groups": int(distinct),
            "duplicate_groups": int(duplicate_groups),
            "affected_rows": int(affected),
            "surplus_rows": int(surplus),
            "maximum_group_size": int(maximum),
            "multiplicity_histogram": {
                str(key): int(value) for key, value in sorted(histogram.items())
            },
        },
        duplicate_keys,
    )


def _key_pass_daily(
    source: _AuthenticatedSource,
    settings: DiscoverySettings,
    partitions: _PackedPartitions,
) -> dict[str, int]:
    full_rows = retained_rows = 0
    with source.batches(("DlyCalDt", "PERMNO"), settings.block_size_bytes) as batches:
        for batch in batches:
            full_rows += len(batch)
            retained, calendar = _scope_by_date(
                batch,
                "DlyCalDt",
                settings.start_date,
                settings.end_date,
                "daily",
            )
            if len(retained) == 0:
                continue
            permnos = _positive_permnos(
                retained.column(retained.schema.get_field_index("PERMNO")),
                "daily retained key pass",
            )
            records = np.empty(len(retained), dtype=_DAILY_DTYPE)
            records["permno"] = permnos
            records["calendar"] = _date_days(calendar)
            partitions.append(records)
            retained_rows += len(records)
    return {"full_file_rows": int(full_rows), "retained_rows": int(retained_rows)}


def _key_pass_monthly(
    source: _AuthenticatedSource,
    settings: DiscoverySettings,
    partitions: _PackedPartitions,
) -> dict[str, int]:
    selected = (
        "MthCalDt",
        "PERMNO",
        "SecInfoStartDt",
        "SecInfoEndDt",
    )
    full_rows = retained_rows = 0
    with source.batches(selected, settings.block_size_bytes) as batches:
        for batch in batches:
            full_rows += len(batch)
            retained, calendar = _scope_by_date(
                batch,
                "MthCalDt",
                settings.start_date,
                settings.end_date,
                "monthly",
            )
            if len(retained) == 0:
                continue
            permnos = _positive_permnos(
                retained.column(retained.schema.get_field_index("PERMNO")),
                "monthly retained key pass",
            )
            starts = _strict_date32(
                retained.column(retained.schema.get_field_index("SecInfoStartDt")),
                "monthly embedded interval start",
            )
            ends = _strict_date32(
                retained.column(retained.schema.get_field_index("SecInfoEndDt")),
                "monthly embedded interval end",
            )
            if bool(
                np.asarray(pc.greater(starts, ends).to_numpy(zero_copy_only=False)).any()
            ):
                raise TypedBackboneDiscoveryError(
                    "Monthly retained rows contain inverted embedded intervals"
                )
            records = np.empty(len(retained), dtype=_MONTHLY_DTYPE)
            records["permno"] = permnos
            records["calendar"] = _date_days(calendar)
            records["interval_start"] = _date_days(starts)
            records["interval_end"] = _date_days(ends)
            partitions.append(records)
            retained_rows += len(records)
    return {"full_file_rows": int(full_rows), "retained_rows": int(retained_rows)}


@dataclasses.dataclass(frozen=True)
class _NameInterval:
    start: int
    end: int
    payload: tuple[str, ...]


@dataclasses.dataclass
class _NamesIndex:
    intervals: dict[int, tuple[_NameInterval, ...]]
    starts: dict[int, tuple[int, ...]]
    exact: dict[tuple[int, int, int], _NameInterval]
    full_file_rows: int
    retained_rows: int
    unique_permnos: int
    contiguous_edges: int
    gap_edges: int
    overlap_edges: int

    def as_of(self, permno: int, date: int) -> tuple[_NameInterval, ...]:
        values = self.intervals.get(permno, ())
        if not values:
            return ()
        starts = self.starts[permno]
        position = bisect.bisect_right(starts, date) - 1
        if position >= 0 and values[position].start <= date <= values[position].end:
            return (values[position],)
        return ()

    def public_summary(self) -> dict[str, Any]:
        return {
            "full_file_rows": self.full_file_rows,
            "retained_rows": self.retained_rows,
            "unique_permnos": self.unique_permnos,
            "duplicate_key_groups": 0,
            "contiguous_edges": self.contiguous_edges,
            "gap_edges": self.gap_edges,
            "overlap_edges": self.overlap_edges,
            "interval_bounds": "inclusive",
        }


def _load_names_index(
    source: _AuthenticatedSource, settings: DiscoverySettings
) -> _NamesIndex:
    header = source.spec.header
    required = {
        "PERMNO",
        "SecInfoStartDt",
        "SecInfoEndDt",
        *IDENTITY_PAYLOAD_FIELDS,
    }
    if not required.issubset(header):
        raise TypedBackboneDiscoveryError("Names fixture lacks identity fields")
    indexes = {column: header.index(column) for column in header}
    grouped: dict[int, list[_NameInterval]] = defaultdict(list)
    exact: dict[tuple[int, int, int], _NameInterval] = {}
    full_rows = retained_rows = 0

    with source.batches(header, settings.block_size_bytes) as batches:
        for batch in batches:
            full_rows += len(batch)
            # Only the two interval endpoints are inspected before scope routing.
            raw_starts = batch.column(batch.schema.get_field_index("SecInfoStartDt"))
            raw_ends = batch.column(batch.schema.get_field_index("SecInfoEndDt"))
            starts_all = _strict_date32(raw_starts, "names scope interval start")
            ends_all = _strict_date32(raw_ends, "names scope interval end")
            inverted = pc.greater(starts_all, ends_all)
            if bool(np.asarray(inverted.to_numpy(zero_copy_only=False)).any()):
                raise TypedBackboneDiscoveryError("Names source contains inverted intervals")
            mask = pc.and_(
                pc.greater_equal(raw_ends, settings.start_date.isoformat()),
                pc.less_equal(raw_starts, settings.end_date.isoformat()),
            )
            retained = _filter(batch, mask)
            if len(retained) == 0:
                continue
            starts = pc.filter(starts_all, mask)
            ends = pc.filter(ends_all, mask)
            permnos = _positive_permnos(
                retained.column(retained.schema.get_field_index("PERMNO")),
                "names retained intervals",
            )
            start_days = _date_days(starts)
            end_days = _date_days(ends)
            rows = _decoded_retained_rows(retained, "names")
            for permno, start, end, row in zip(
                permnos.tolist(),
                start_days.tolist(),
                end_days.tolist(),
                rows,
                strict=True,
            ):
                key = (int(permno), int(start), int(end))
                payload = tuple(row[indexes[field]] for field in IDENTITY_PAYLOAD_FIELDS)
                interval = _NameInterval(int(start), int(end), payload)
                if key in exact:
                    # Supporting-source policy makes even an exact duplicate fatal.
                    raise TypedBackboneDiscoveryError(
                        "Names interval key is duplicated in retained scope"
                    )
                exact[key] = interval
                grouped[int(permno)].append(interval)
                retained_rows += 1

    intervals: dict[int, tuple[_NameInterval, ...]] = {}
    starts_lookup: dict[int, tuple[int, ...]] = {}
    contiguous = gaps = overlaps = 0
    for permno, values in grouped.items():
        ordered = tuple(sorted(values, key=lambda value: (value.start, value.end)))
        for previous, current in zip(ordered, ordered[1:]):
            if current.start <= previous.end:
                overlaps += 1
            elif current.start == previous.end + 1:
                contiguous += 1
            else:
                gaps += 1
        intervals[permno] = ordered
        starts_lookup[permno] = tuple(value.start for value in ordered)
    if overlaps:
        raise TypedBackboneDiscoveryError(
            f"Names intervals are not nonoverlapping (overlap_edges={overlaps})"
        )
    return _NamesIndex(
        intervals=intervals,
        starts=starts_lookup,
        exact=exact,
        full_file_rows=int(full_rows),
        retained_rows=int(retained_rows),
        unique_permnos=len(intervals),
        contiguous_edges=int(contiguous),
        gap_edges=int(gaps),
        overlap_edges=int(overlaps),
    )


def _blank_field_counts() -> dict[str, dict[str, int]]:
    return {
        field: {"compared_rows": 0, "equal_rows": 0, "different_rows": 0}
        for field in IDENTITY_PAYLOAD_FIELDS
    }


@dataclasses.dataclass
class _IdentityDiagnostics:
    source: str
    as_of_date_field: str
    rows: int = 0
    blank_as_of_date_rows: int = 0
    as_of_zero: int = 0
    as_of_one: int = 0
    as_of_multiple: int = 0
    exact_key_match: int = 0
    exact_key_missing: int = 0
    as_of_interval_key_equal: int = 0
    as_of_interval_key_different: int = 0
    exact_key_all_payload_equal: int = 0
    exact_key_payload_different: int = 0
    as_of_all_payload_equal: int = 0
    as_of_payload_different: int = 0
    exact_fields: dict[str, dict[str, int]] = dataclasses.field(
        default_factory=_blank_field_counts
    )
    as_of_fields: dict[str, dict[str, int]] = dataclasses.field(
        default_factory=_blank_field_counts
    )

    @staticmethod
    def _compare(
        row: tuple[str, ...],
        indexes: Mapping[str, int],
        interval: _NameInterval,
        counters: dict[str, dict[str, int]],
    ) -> bool:
        all_equal = True
        for position, field in enumerate(IDENTITY_PAYLOAD_FIELDS):
            equal = row[indexes[field]] == interval.payload[position]
            counters[field]["compared_rows"] += 1
            counters[field]["equal_rows"] += int(equal)
            counters[field]["different_rows"] += int(not equal)
            all_equal &= equal
        return all_equal

    def observe(
        self,
        *,
        permno: int,
        embedded_start: int,
        embedded_end: int,
        as_of_date: int | None,
        row: tuple[str, ...],
        indexes: Mapping[str, int],
        names: _NamesIndex,
    ) -> None:
        self.rows += 1
        exact = names.exact.get((permno, embedded_start, embedded_end))
        if exact is None:
            self.exact_key_missing += 1
        else:
            self.exact_key_match += 1
            if self._compare(row, indexes, exact, self.exact_fields):
                self.exact_key_all_payload_equal += 1
            else:
                self.exact_key_payload_different += 1

        if as_of_date is None:
            self.blank_as_of_date_rows += 1
            matches: tuple[_NameInterval, ...] = ()
        else:
            matches = names.as_of(permno, as_of_date)
        if not matches:
            self.as_of_zero += 1
            return
        if len(matches) > 1:
            self.as_of_multiple += 1
            return
        self.as_of_one += 1
        selected = matches[0]
        same_key = selected.start == embedded_start and selected.end == embedded_end
        self.as_of_interval_key_equal += int(same_key)
        self.as_of_interval_key_different += int(not same_key)
        if self._compare(row, indexes, selected, self.as_of_fields):
            self.as_of_all_payload_equal += 1
        else:
            self.as_of_payload_different += 1

    def public(self) -> dict[str, Any]:
        if self.rows != self.as_of_zero + self.as_of_one + self.as_of_multiple:
            raise TypedBackboneDiscoveryError(
                f"{self.source} as-of match-count conservation failed"
            )
        if self.rows != self.exact_key_match + self.exact_key_missing:
            raise TypedBackboneDiscoveryError(
                f"{self.source} exact-key conservation failed"
            )
        return {
            "source": self.source,
            "canonical_rows": self.rows,
            "as_of_date_field": self.as_of_date_field,
            "interval_bounds": "inclusive",
            "blank_as_of_date_rows": self.blank_as_of_date_rows,
            "as_of_match_counts": {
                "zero": self.as_of_zero,
                "one": self.as_of_one,
                "multiple": self.as_of_multiple,
            },
            "embedded_interval_exact_key": {
                "match_rows": self.exact_key_match,
                "missing_rows": self.exact_key_missing,
            },
            "as_of_interval_key_agreement": {
                "equal_rows": self.as_of_interval_key_equal,
                "different_rows": self.as_of_interval_key_different,
            },
            "exact_key_payload": {
                "all_fields_equal_rows": self.exact_key_all_payload_equal,
                "one_or_more_fields_different_rows": self.exact_key_payload_different,
                "field_counts": self.exact_fields,
            },
            "as_of_payload": {
                "all_fields_equal_rows": self.as_of_all_payload_equal,
                "one_or_more_fields_different_rows": self.as_of_payload_different,
                "field_counts": self.as_of_fields,
            },
        }


def _required_identity_header(header: Sequence[str], source: str) -> dict[str, int]:
    required = {
        "PERMNO",
        "SecInfoStartDt",
        "SecInfoEndDt",
        *IDENTITY_PAYLOAD_FIELDS,
    }
    if source == "daily":
        required.add("DlyCalDt")
    elif source == "monthly":
        required.update(("MthCalDt", "MthPrcDt"))
    missing = sorted(required - set(header))
    if missing:
        raise TypedBackboneDiscoveryError(
            f"{source} schema lacks required identity discovery columns: {missing}"
        )
    return {column: header.index(column) for column in header}


def _scan_daily_canonical(
    source: _AuthenticatedSource,
    settings: DiscoverySettings,
    names: _NamesIndex,
    duplicate_keys: Mapping[tuple[int, int], int],
) -> tuple[dict[str, Any], _IdentityDiagnostics]:
    header = source.spec.header
    indexes = _required_identity_header(header, "daily")
    diagnostics = _IdentityDiagnostics("daily", "DlyCalDt")
    duplicate_first: dict[tuple[int, int], tuple[str, ...]] = {}
    duplicate_seen: Counter[tuple[int, int]] = Counter()
    full_rows = retained_rows = canonical_rows = 0

    with source.batches(header, settings.block_size_bytes) as batches:
        for batch in batches:
            full_rows += len(batch)
            retained, calendar = _scope_by_date(
                batch,
                "DlyCalDt",
                settings.start_date,
                settings.end_date,
                "daily canonical replay",
            )
            if len(retained) == 0:
                continue
            permnos = _positive_permnos(
                retained.column(retained.schema.get_field_index("PERMNO")),
                "daily canonical replay",
            )
            starts_array = _strict_date32(
                retained.column(retained.schema.get_field_index("SecInfoStartDt")),
                "daily embedded interval start",
            )
            ends_array = _strict_date32(
                retained.column(retained.schema.get_field_index("SecInfoEndDt")),
                "daily embedded interval end",
            )
            if bool(
                np.asarray(
                    pc.greater(starts_array, ends_array).to_numpy(zero_copy_only=False)
                ).any()
            ):
                raise TypedBackboneDiscoveryError(
                    "Daily retained rows contain inverted embedded intervals"
                )
            calendars = _date_days(calendar)
            starts = _date_days(starts_array)
            ends = _date_days(ends_array)
            rows = _decoded_retained_rows(retained, "daily")
            for permno, date, start, end, row in zip(
                permnos.tolist(),
                calendars.tolist(),
                starts.tolist(),
                ends.tolist(),
                rows,
                strict=True,
            ):
                key = (int(permno), int(date))
                if key in duplicate_keys:
                    duplicate_seen[key] += 1
                    first = duplicate_first.setdefault(key, row)
                    if row != first:
                        raise TypedBackboneDiscoveryError(
                            "Daily duplicate key has conflicting decoded payloads"
                        )
                    if duplicate_seen[key] > 1:
                        retained_rows += 1
                        continue
                diagnostics.observe(
                    permno=int(permno),
                    embedded_start=int(start),
                    embedded_end=int(end),
                    as_of_date=int(date),
                    row=row,
                    indexes=indexes,
                    names=names,
                )
                canonical_rows += 1
                retained_rows += 1

    if dict(duplicate_seen) != dict(duplicate_keys):
        raise TypedBackboneDiscoveryError(
            "Daily duplicate replay multiplicities differ from key pass"
        )
    return (
        {
            "full_file_rows": int(full_rows),
            "retained_rows": int(retained_rows),
            "canonical_rows": int(canonical_rows),
            "all_field_exact_duplicate_groups": len(duplicate_keys),
            "conflicting_duplicate_groups": 0,
        },
        diagnostics,
    )


@dataclasses.dataclass(frozen=True)
class _MonthlyRow:
    row: tuple[str, ...]
    permno: int
    calendar: int
    interval_start: int
    interval_end: int
    price_date: int | None


def _scan_monthly_canonical(
    source: _AuthenticatedSource,
    settings: DiscoverySettings,
    names: _NamesIndex,
    raw_duplicate_keys: Mapping[tuple[int, int, int, int], int],
    candidate_keys: Mapping[tuple[int, int], int],
) -> tuple[dict[str, Any], _IdentityDiagnostics]:
    header = source.spec.header
    indexes = _required_identity_header(header, "monthly")
    diagnostics = _IdentityDiagnostics("monthly", "MthPrcDt")
    duplicate_first: dict[tuple[int, int, int, int], tuple[str, ...]] = {}
    duplicate_seen: Counter[tuple[int, int, int, int]] = Counter()
    candidates: dict[tuple[int, int], list[_MonthlyRow]] = defaultdict(list)
    full_rows = retained_rows = canonical_rows = 0
    blank_price_dates = price_date_month_mismatches = 0

    def consume(value: _MonthlyRow) -> None:
        nonlocal canonical_rows, blank_price_dates, price_date_month_mismatches
        if value.price_date is None:
            blank_price_dates += 1
        else:
            calendar_date = _EPOCH + dt.timedelta(days=value.calendar)
            price_date = _EPOCH + dt.timedelta(days=value.price_date)
            if (calendar_date.year, calendar_date.month) != (
                price_date.year,
                price_date.month,
            ):
                price_date_month_mismatches += 1
        diagnostics.observe(
            permno=value.permno,
            embedded_start=value.interval_start,
            embedded_end=value.interval_end,
            as_of_date=value.price_date,
            row=value.row,
            indexes=indexes,
            names=names,
        )
        canonical_rows += 1

    with source.batches(header, settings.block_size_bytes) as batches:
        for batch in batches:
            full_rows += len(batch)
            retained, calendar = _scope_by_date(
                batch,
                "MthCalDt",
                settings.start_date,
                settings.end_date,
                "monthly canonical replay",
            )
            if len(retained) == 0:
                continue
            permnos = _positive_permnos(
                retained.column(retained.schema.get_field_index("PERMNO")),
                "monthly canonical replay",
            )
            starts_array = _strict_date32(
                retained.column(retained.schema.get_field_index("SecInfoStartDt")),
                "monthly embedded interval start",
            )
            ends_array = _strict_date32(
                retained.column(retained.schema.get_field_index("SecInfoEndDt")),
                "monthly embedded interval end",
            )
            if bool(
                np.asarray(
                    pc.greater(starts_array, ends_array).to_numpy(zero_copy_only=False)
                ).any()
            ):
                raise TypedBackboneDiscoveryError(
                    "Monthly retained rows contain inverted embedded intervals"
                )
            calendars = _date_days(calendar)
            starts = _date_days(starts_array)
            ends = _date_days(ends_array)
            price_dates = _optional_date_days(
                retained.column(retained.schema.get_field_index("MthPrcDt")),
                "monthly price date",
            )
            rows = _decoded_retained_rows(retained, "monthly")
            for permno, cal, start, end, price_date, row in zip(
                permnos.tolist(),
                calendars.tolist(),
                starts.tolist(),
                ends.tolist(),
                price_dates,
                rows,
                strict=True,
            ):
                raw_key = (int(permno), int(cal), int(start), int(end))
                if raw_key in raw_duplicate_keys:
                    duplicate_seen[raw_key] += 1
                    first = duplicate_first.setdefault(raw_key, row)
                    if row != first:
                        raise TypedBackboneDiscoveryError(
                            "Monthly raw duplicate key has conflicting decoded payloads"
                        )
                    if duplicate_seen[raw_key] > 1:
                        retained_rows += 1
                        continue
                value = _MonthlyRow(
                    row=row,
                    permno=int(permno),
                    calendar=int(cal),
                    interval_start=int(start),
                    interval_end=int(end),
                    price_date=price_date,
                )
                canonical_key = (value.permno, value.calendar)
                if canonical_key in candidate_keys:
                    candidates[canonical_key].append(value)
                else:
                    consume(value)
                retained_rows += 1

    if dict(duplicate_seen) != dict(raw_duplicate_keys):
        raise TypedBackboneDiscoveryError(
            "Monthly duplicate replay multiplicities differ from key pass"
        )
    expected_candidates = dict(candidate_keys)
    if set(candidates) != set(expected_candidates):
        raise TypedBackboneDiscoveryError(
            "Monthly interval candidate keys differ from key pass"
        )

    price_zero = price_one = price_multiple = 0
    calendar_zero = calendar_one = calendar_multiple = 0
    selected_different = 0
    candidate_size_histogram: Counter[int] = Counter()
    for key in sorted(candidates):
        values = sorted(
            candidates[key],
            key=lambda value: (
                value.interval_start,
                value.interval_end,
                tuple(field.encode("utf-8") for field in value.row),
            ),
        )
        if len(values) != expected_candidates[key]:
            raise TypedBackboneDiscoveryError(
                "Monthly candidate multiplicity differs after exact collapse"
            )
        locked_candidate_size = settings.sources["monthly"].candidate_group_size
        if not settings.fixture_mode and len(values) != locked_candidate_size:
            raise TypedBackboneDiscoveryError(
                "Monthly candidate group size differs from the exact locked size"
            )
        candidate_size_histogram[len(values)] += 1
        price_matches = [
            index
            for index, value in enumerate(values)
            if value.price_date is not None
            and value.interval_start <= value.price_date <= value.interval_end
        ]
        calendar_matches = [
            index
            for index, value in enumerate(values)
            if value.interval_start <= value.calendar <= value.interval_end
        ]
        price_zero += int(len(price_matches) == 0)
        price_one += int(len(price_matches) == 1)
        price_multiple += int(len(price_matches) > 1)
        calendar_zero += int(len(calendar_matches) == 0)
        calendar_one += int(len(calendar_matches) == 1)
        calendar_multiple += int(len(calendar_matches) > 1)
        if len(price_matches) != 1 or len(calendar_matches) != 1:
            raise TypedBackboneDiscoveryError(
                "Monthly candidate does not have unique price/calendar interval matches"
            )
        if price_matches[0] != calendar_matches[0]:
            selected_different += 1
            raise TypedBackboneDiscoveryError(
                "Monthly price and calendar dates select different interval rows"
            )
        consume(values[price_matches[0]])

    return (
        {
            "full_file_rows": int(full_rows),
            "retained_rows": int(retained_rows),
            "canonical_rows": int(canonical_rows),
            "all_field_exact_raw_duplicate_groups": len(raw_duplicate_keys),
            "conflicting_raw_duplicate_groups": 0,
            "interval_candidate_groups": len(candidates),
            "candidate_size_histogram": {
                str(key): int(value)
                for key, value in sorted(candidate_size_histogram.items())
            },
            "candidate_selection": {
                "price_date_zero_matches": int(price_zero),
                "price_date_one_match": int(price_one),
                "price_date_multiple_matches": int(price_multiple),
                "calendar_date_zero_matches": int(calendar_zero),
                "calendar_date_one_match": int(calendar_one),
                "calendar_date_multiple_matches": int(calendar_multiple),
                "price_and_calendar_selected_different_rows": int(selected_different),
            },
            "observation_date": {
                "field": "MthPrcDt",
                "blank_rows": int(blank_price_dates),
                "nonblank_rows": int(canonical_rows - blank_price_dates),
                "outside_MthCalDt_calendar_month_rows": int(
                    price_date_month_mismatches
                ),
                "MthCalDt_substitution_permitted": False,
            },
        },
        diagnostics,
    )


def _assert_expected(
    settings: DiscoverySettings,
    *,
    daily_key: Mapping[str, Any],
    daily_scan: Mapping[str, Any],
    monthly_key: Mapping[str, Any],
    monthly_raw: Mapping[str, Any],
    monthly_post: Mapping[str, Any],
    monthly_scan: Mapping[str, Any],
    names: _NamesIndex,
) -> None:
    if settings.fixture_mode:
        fixture_passes = {
            "daily": (
                int(daily_key["full_file_rows"]),
                int(daily_scan["full_file_rows"]),
            ),
            "monthly": (
                int(monthly_key["full_file_rows"]),
                int(monthly_scan["full_file_rows"]),
            ),
            "names": (int(names.full_file_rows),),
        }
        for source_name, observed_passes in fixture_passes.items():
            locked_rows = settings.sources[source_name].full_file_rows
            if type(locked_rows) is not int:
                raise TypedBackboneDiscoveryError(
                    f"Fixture {source_name} full-file row lock is absent"
                )
            for observed_rows in observed_passes:
                if observed_rows > FIXTURE_MAX_ROWS_PER_SOURCE:
                    raise TypedBackboneDiscoveryError(
                        f"Fixture {source_name} observed full rows exceed the ceiling"
                    )
                if observed_rows != locked_rows:
                    raise TypedBackboneDiscoveryError(
                        f"Fixture {source_name} observed full rows differ from lock"
                    )
        return
    actual = {
        "daily": {
            "full_file_rows": daily_key["full_file_rows"],
            "retained_rows": daily_key["retained_rows"],
            "canonical_rows": daily_scan["canonical_rows"],
            "duplicate_groups": daily_key["duplicate_groups"],
            "duplicate_surplus_rows": daily_key["surplus_rows"],
            "conflicting_groups": daily_scan["conflicting_duplicate_groups"],
        },
        "monthly": {
            "full_file_rows": monthly_key["full_file_rows"],
            "retained_rows": monthly_key["retained_rows"],
            "raw_distinct_rows": monthly_raw["distinct_groups"],
            "canonical_rows": monthly_scan["canonical_rows"],
            "duplicate_groups": monthly_raw["duplicate_groups"],
            "duplicate_surplus_rows": monthly_raw["surplus_rows"],
            "conflicting_groups": monthly_scan["conflicting_raw_duplicate_groups"],
            "interval_candidate_groups": monthly_post["duplicate_groups"],
            "candidate_group_size": monthly_post["maximum_group_size"],
            "candidate_size_histogram": dict(
                monthly_scan["candidate_size_histogram"]
            ),
            "groups_with_exactly_one_price_date_match": monthly_scan[
                "candidate_selection"
            ]["price_date_one_match"],
            "groups_with_zero_price_date_matches": monthly_scan[
                "candidate_selection"
            ]["price_date_zero_matches"],
            "groups_with_multiple_price_date_matches": monthly_scan[
                "candidate_selection"
            ]["price_date_multiple_matches"],
            "groups_with_exactly_one_calendar_date_match": monthly_scan[
                "candidate_selection"
            ]["calendar_date_one_match"],
            "groups_with_zero_calendar_date_matches": monthly_scan[
                "candidate_selection"
            ]["calendar_date_zero_matches"],
            "groups_with_multiple_calendar_date_matches": monthly_scan[
                "candidate_selection"
            ]["calendar_date_multiple_matches"],
        },
        "names": {
            "full_file_rows": names.full_file_rows,
            "retained_rows": names.retained_rows,
            "unique_permnos": names.unique_permnos,
            "contiguous_edges": names.contiguous_edges,
            "gap_edges": names.gap_edges,
            "overlap_edges": names.overlap_edges,
        },
    }
    for source_name, expected in settings.expected.items():
        observed = actual[source_name]
        for field, value in expected.items():
            expected_value = dict(value) if type(value) is dict else value
            if observed.get(field) != expected_value:
                raise TypedBackboneDiscoveryError(
                    f"Locked {source_name} expectation differs for {field}"
                )


def _prepare_scratch_root(value: Path | None) -> Path | None:
    if value is None:
        # Resolve only the OS-selected temp root.  Explicit caller paths retain
        # strict lexical no-symlink enforcement below.
        try:
            root = Path(tempfile.gettempdir()).resolve(strict=True)
        except OSError as exc:
            raise TypedBackboneDiscoveryError(
                "Cannot canonicalize the OS-selected scratch root"
            ) from exc
    else:
        root = _absolute(value)
    if not root.exists():
        raise TypedBackboneDiscoveryError(
            "Private scratch root must already exist; discovery will not create it"
        )
    _assert_no_symlink_components(root, "scratch root")
    metadata = root.lstat()
    if not stat.S_ISDIR(metadata.st_mode):
        raise TypedBackboneDiscoveryError("Private scratch root is not a directory")
    mode = stat.S_IMODE(metadata.st_mode)
    owner_private = (
        metadata.st_uid == os.geteuid()
        and not bool(mode & (stat.S_IWGRP | stat.S_IWOTH))
        and bool(mode & stat.S_IWUSR)
        and bool(mode & stat.S_IXUSR)
    )
    sticky_world_writable = (
        bool(mode & stat.S_IWOTH)
        and bool(mode & stat.S_ISVTX)
        and os.access(root, os.W_OK | os.X_OK)
    )
    if not (owner_private or sticky_world_writable):
        raise TypedBackboneDiscoveryError(
            "Private scratch root lacks safe owner/private or sticky-temp semantics"
        )
    return root


def _freeze_production_value(value: Any) -> Any:
    """Create a type-sensitive snapshot of trusted settings containers."""

    if type(value) is dict:
        frozen: list[tuple[str, Any]] = []
        for key in sorted(value):
            if type(key) is not str:
                raise TypedBackboneDiscoveryError(
                    "Production settings contain a non-string mapping key"
                )
            frozen.append((key, _freeze_production_value(value[key])))
        return ("dict", tuple(frozen))
    if type(value) is tuple:
        return ("tuple", tuple(_freeze_production_value(item) for item in value))
    if type(value) is SourceSpec:
        return (
            "SourceSpec",
            tuple(
                (field.name, _freeze_production_value(getattr(value, field.name)))
                for field in dataclasses.fields(SourceSpec)
            ),
        )
    if isinstance(value, Path):
        return ("path", os.fspath(value))
    if type(value) is dt.date:
        return ("date", value.isoformat())
    if value is None:
        return ("none",)
    if type(value) in {bool, int, str}:
        return (type(value).__name__, value)
    raise TypedBackboneDiscoveryError(
        "Production settings contain a noncanonical value type"
    )


def _assert_production_settings_match(
    provided: DiscoverySettings,
    locked: DiscoverySettings,
) -> None:
    """Compare every production semantic field with the authenticated lock."""

    if type(provided) is not DiscoverySettings or type(locked) is not DiscoverySettings:
        raise TypedBackboneDiscoveryError(
            "Production settings must use the exact DiscoverySettings type"
        )
    if type(provided.expected) is not dict or set(provided.expected) != set(SOURCE_ORDER):
        raise TypedBackboneDiscoveryError(
            "Production expectations must be exact and nonempty"
        )
    if any(type(value) is not dict or not value for value in provided.expected.values()):
        raise TypedBackboneDiscoveryError(
            "Production expectations must be exact and nonempty"
        )
    normalized_raw_root = _absolute(provided.raw_root)
    for field in dataclasses.fields(DiscoverySettings):
        if field.name in {"scratch_root", "block_size_bytes"}:
            continue
        provided_value = getattr(provided, field.name)
        if field.name == "raw_root":
            provided_value = normalized_raw_root
        if _freeze_production_value(provided_value) != _freeze_production_value(
            getattr(locked, field.name)
        ):
            raise TypedBackboneDiscoveryError(
                f"Production setting {field.name} differs from authenticated lock"
            )


def _assert_fixture_settings(settings: DiscoverySettings) -> None:
    """Authenticate a deliberately small, pre-validation-only test fixture."""

    if type(settings) is not DiscoverySettings or settings.fixture_mode is not True:
        raise TypedBackboneDiscoveryError(
            "Fixture settings must use the exact explicit fixture contract"
        )
    if type(settings.start_date) is not dt.date or type(settings.end_date) is not dt.date:
        raise TypedBackboneDiscoveryError("Fixture scope dates must be exact dates")
    if not (
        FIXTURE_SCOPE_START
        <= settings.start_date
        <= settings.end_date
        <= FIXTURE_SCOPE_END
    ):
        raise TypedBackboneDiscoveryError(
            "Fixture scope must remain inside the authorized quality envelope"
        )
    if type(settings.block_size_bytes) is not int or not (
        _MIN_ARROW_BLOCK_SIZE <= settings.block_size_bytes <= FIXTURE_MAX_SOURCE_BYTES
    ):
        raise TypedBackboneDiscoveryError(
            "Fixture block size is outside the strict small-fixture range"
        )
    for name, value in (
        ("expected", settings.expected),
        ("production_config_sha256", settings.production_config_sha256),
        (
            "production_identity_count_status",
            settings.production_identity_count_status,
        ),
    ):
        if type(value) is not dict or value:
            raise TypedBackboneDiscoveryError(
                f"Fixture {name} must be an exact empty dict"
            )
    if type(settings.sources) is not dict or set(settings.sources) != set(SOURCE_ORDER):
        raise TypedBackboneDiscoveryError(
            "Fixture sources must be an exact daily/monthly/names dict"
        )

    locked = load_production_settings()
    fixture_root = _absolute(settings.raw_root)
    if fixture_root == locked.raw_root:
        raise TypedBackboneDiscoveryError(
            "Fixture mode may not address the authenticated production raw root"
        )
    locked_hashes = {spec.sha256 for spec in locked.sources.values()}
    locked_name_sizes = {
        (spec.filename, spec.size_bytes) for spec in locked.sources.values()
    }
    total_bytes = 0
    total_full_rows = 0
    row_fields = (
        "retained_rows",
        "canonical_rows",
        "duplicate_groups",
        "duplicate_surplus_rows",
        "interval_candidate_groups",
        "candidate_group_size",
    )
    for source_name in SOURCE_ORDER:
        spec = settings.sources[source_name]
        if type(spec) is not SourceSpec:
            raise TypedBackboneDiscoveryError(
                "Fixture source contracts must use the exact SourceSpec type"
            )
        if spec.name != source_name:
            raise TypedBackboneDiscoveryError(
                "Fixture source mapping and SourceSpec name differ"
            )
        if (
            type(spec.filename) is not str
            or Path(spec.filename).name != spec.filename
            or "/" in spec.filename
            or "\\" in spec.filename
            or spec.filename in {"", ".", ".."}
        ):
            raise TypedBackboneDiscoveryError("Fixture filename is not a direct child")
        if (
            type(spec.header) is not tuple
            or not spec.header
            or any(type(column) is not str for column in spec.header)
            or len(set(spec.header)) != len(spec.header)
        ):
            raise TypedBackboneDiscoveryError("Fixture header contract is invalid")
        if (
            type(spec.sha256) is not str
            or len(spec.sha256) != 64
            or any(character not in "0123456789abcdef" for character in spec.sha256)
        ):
            raise TypedBackboneDiscoveryError("Fixture SHA-256 contract is invalid")
        if type(spec.size_bytes) is not int or not (
            0 < spec.size_bytes <= FIXTURE_MAX_SOURCE_BYTES
        ):
            raise TypedBackboneDiscoveryError(
                "Fixture source exceeds the strict per-source byte ceiling"
            )
        total_bytes += spec.size_bytes
        if spec.sha256 in locked_hashes or (
            spec.filename,
            spec.size_bytes,
        ) in locked_name_sizes:
            raise TypedBackboneDiscoveryError(
                "Fixture source matches an authenticated production source identity"
            )
        if type(spec.full_file_rows) is not int or not (
            0 <= spec.full_file_rows <= FIXTURE_MAX_ROWS_PER_SOURCE
        ):
            raise TypedBackboneDiscoveryError(
                "Fixture full_file_rows must be an exact non-null count within ceiling"
            )
        for field in row_fields:
            count = getattr(spec, field)
            if count is None:
                continue
            if type(count) is not int or not (
                0 <= count <= FIXTURE_MAX_ROWS_PER_SOURCE
            ):
                raise TypedBackboneDiscoveryError(
                    f"Fixture {field} exceeds the strict row ceiling"
                )
        total_full_rows += spec.full_file_rows
    if total_bytes > FIXTURE_MAX_TOTAL_BYTES:
        raise TypedBackboneDiscoveryError(
            "Fixture sources exceed the strict total byte ceiling"
        )
    if total_full_rows > FIXTURE_MAX_TOTAL_FULL_FILE_ROWS:
        raise TypedBackboneDiscoveryError(
            "Fixture sources exceed the strict total row ceiling"
        )


def _authenticate_discovery_settings(
    settings: DiscoverySettings | None,
) -> DiscoverySettings:
    """Select explicit fixture mode or re-derive an immutable production contract."""

    if settings is None:
        return load_production_settings()
    if type(settings) is not DiscoverySettings:
        raise TypedBackboneDiscoveryError(
            "Discovery settings must use the exact DiscoverySettings type"
        )
    if type(settings.fixture_mode) is not bool:
        raise TypedBackboneDiscoveryError("Discovery mode flag must be an exact boolean")
    if settings.fixture_mode is True:
        _assert_fixture_settings(settings)
        return settings
    if type(settings.block_size_bytes) is not int or settings.block_size_bytes <= 0:
        raise TypedBackboneDiscoveryError(
            "Production discovery block size must be a positive integer"
        )
    locked = load_production_settings(
        scratch_root=settings.scratch_root,
        block_size_bytes=settings.block_size_bytes,
    )
    _assert_production_settings_match(settings, locked)
    # Discard all caller-owned production containers after comparison.  The
    # scanner consumes only the newly authenticated locked reconstruction.
    return locked


def discover_typed_backbone_identity_joins(
    settings: DiscoverySettings | None = None,
) -> dict[str, Any]:
    """Return deterministic aggregate discovery evidence and write nothing durable."""

    settings = _authenticate_discovery_settings(settings)
    if set(settings.sources) != set(SOURCE_ORDER):
        raise TypedBackboneDiscoveryError(
            "Discovery sources must be exactly daily, monthly, and names"
        )
    if not settings.fixture_mode and set(settings.expected) != set(SOURCE_ORDER):
        raise TypedBackboneDiscoveryError(
            "Production discovery requires all locked row-count expectations"
        )
    if not settings.fixture_mode:
        if dict(settings.production_config_sha256) != _locked_production_config_sha256():
            raise TypedBackboneDiscoveryError(
                "Production discovery config bindings are incomplete or differ"
            )
        expected_stage = {
            "daily": PENDING_IDENTITY_COUNT_STATUS,
            "monthly": PENDING_IDENTITY_COUNT_STATUS,
        }
        if dict(settings.production_identity_count_status) != expected_stage:
            raise TypedBackboneDiscoveryError(
                "Production identity-count discovery stage is no longer pending"
            )
    for name in SOURCE_ORDER:
        spec = settings.sources[name]
        if spec.name != name:
            raise TypedBackboneDiscoveryError("Source mapping and source name differ")
        if (
            len(spec.sha256) != 64
            or any(character not in "0123456789abcdef" for character in spec.sha256)
            or spec.size_bytes < 0
        ):
            raise TypedBackboneDiscoveryError("Source authentication contract is invalid")
    if settings.start_date > settings.end_date:
        raise TypedBackboneDiscoveryError("Discovery scope is inverted")
    if settings.block_size_bytes < _MIN_ARROW_BLOCK_SIZE:
        raise TypedBackboneDiscoveryError("Arrow block size is below safe minimum")
    scratch_parent = _prepare_scratch_root(settings.scratch_root)

    raw_root = _absolute(settings.raw_root)
    _assert_no_symlink_components(raw_root, "raw root")
    with ExitStack() as stack:
        opened = {
            name: stack.enter_context(
                _AuthenticatedSource.open(raw_root, settings.sources[name])
            )
            for name in SOURCE_ORDER
        }
        with tempfile.TemporaryDirectory(
            prefix=".crsp-typed-identity-discovery-", dir=scratch_parent
        ) as temporary:
            private = Path(temporary)
            private.chmod(0o700)
            daily_partitions = _PackedPartitions(private / "daily", _DAILY_DTYPE)
            monthly_partitions = _PackedPartitions(private / "monthly", _MONTHLY_DTYPE)

            daily_key_counts = _key_pass_daily(
                opened["daily"], settings, daily_partitions
            )
            monthly_key_counts = _key_pass_monthly(
                opened["monthly"], settings, monthly_partitions
            )
            daily_key_metrics, daily_duplicates = _summarize_partitions(
                daily_partitions, ("permno", "calendar")
            )
            monthly_raw_metrics, monthly_raw_duplicates = _summarize_partitions(
                monthly_partitions,
                ("permno", "calendar", "interval_start", "interval_end"),
            )
            monthly_post_metrics, monthly_candidates = _summarize_partitions(
                monthly_partitions,
                ("permno", "calendar"),
                collapse_raw_monthly_first=True,
            )
            names = _load_names_index(opened["names"], settings)
            daily_scan, daily_identity = _scan_daily_canonical(
                opened["daily"], settings, names, daily_duplicates
            )
            monthly_scan, monthly_identity = _scan_monthly_canonical(
                opened["monthly"],
                settings,
                names,
                monthly_raw_duplicates,
                monthly_candidates,
            )

            if daily_scan["full_file_rows"] != daily_key_counts["full_file_rows"]:
                raise TypedBackboneDiscoveryError("Daily replay full-row count differs")
            if daily_scan["retained_rows"] != daily_key_counts["retained_rows"]:
                raise TypedBackboneDiscoveryError("Daily replay retained count differs")
            if monthly_scan["full_file_rows"] != monthly_key_counts["full_file_rows"]:
                raise TypedBackboneDiscoveryError("Monthly replay full-row count differs")
            if monthly_scan["retained_rows"] != monthly_key_counts["retained_rows"]:
                raise TypedBackboneDiscoveryError("Monthly replay retained count differs")
            if daily_scan["canonical_rows"] != daily_key_metrics["distinct_groups"]:
                raise TypedBackboneDiscoveryError("Daily canonical row conservation failed")
            expected_monthly_canonical = (
                monthly_raw_metrics["distinct_groups"]
                - monthly_post_metrics["surplus_rows"]
            )
            if monthly_scan["canonical_rows"] != expected_monthly_canonical:
                raise TypedBackboneDiscoveryError("Monthly canonical row conservation failed")
            _assert_expected(
                settings,
                daily_key={**daily_key_counts, **daily_key_metrics},
                daily_scan=daily_scan,
                monthly_key=monthly_key_counts,
                monthly_raw=monthly_raw_metrics,
                monthly_post=monthly_post_metrics,
                monthly_scan=monthly_scan,
                names=names,
            )
            for source in opened.values():
                source.assert_stable()

            result = {
                "schema_version": 1,
                "status": "aggregate_identity_join_discovery_complete_not_authoritative",
                "mode": "fixture" if settings.fixture_mode else "production_discovery",
                "contract": {
                    "identity_payload_fields": list(IDENTITY_PAYLOAD_FIELDS),
                    "daily_as_of_date_field": "DlyCalDt",
                    "monthly_as_of_date_field": "MthPrcDt",
                    "monthly_calendar_date_substitution_permitted": False,
                    "decoded_string_normalization_permitted": False,
                    "production_development_config_sha256": (
                        None
                        if settings.fixture_mode
                        else LOCKED_DEVELOPMENT_CONFIG_SHA256
                    ),
                    "production_supporting_config_sha256": (
                        None
                        if settings.fixture_mode
                        else LOCKED_SUPPORTING_CONFIG_SHA256
                    ),
                    "production_typed_policy_config_sha256": (
                        None
                        if settings.fixture_mode
                        else settings.production_config_sha256["typed_policy"]
                    ),
                    "production_typed_schema_config_sha256": (
                        None
                        if settings.fixture_mode
                        else settings.production_config_sha256["typed_schema"]
                    ),
                    "production_paths_config_sha256": (
                        None
                        if settings.fixture_mode
                        else settings.production_config_sha256["paths"]
                    ),
                    "production_database_relocation_receipt_sha256": (
                        None
                        if settings.fixture_mode
                        else settings.production_config_sha256["database_relocation"]
                    ),
                    "production_identity_count_status_at_start": (
                        None
                        if settings.fixture_mode
                        else dict(settings.production_identity_count_status)
                    ),
                    "live_upstream_artifact_pointers_authenticated": False,
                    "checked_in_config_authentication_grants_provenance_authority": False,
                },
                "scope": {
                    "start_date": settings.start_date.isoformat(),
                    "end_date": settings.end_date.isoformat(),
                    "date_bounds_are_inclusive": True,
                    "date_first_payload_routing": True,
                    "excluded_non_date_payload_aggregated": False,
                },
                "authentication": {
                    name: {
                        "filename": opened[name].spec.filename,
                        "sha256": opened[name].sha256,
                        "size_bytes": int(opened[name].opened.st_size),
                        "stable_identity_fields_checked": [
                            "device",
                            "inode",
                            "size_bytes",
                            "mtime_ns",
                            "ctime_ns",
                        ],
                        "nofollow_anchor_held_for_entire_scan": True,
                    }
                    for name in SOURCE_ORDER
                },
                "names": names.public_summary(),
                "daily": {
                    **daily_scan,
                    "key_profile": daily_key_metrics,
                    "identity": daily_identity.public(),
                    "canonicalization": "global exact all-field duplicate collapse",
                },
                "monthly": {
                    **monthly_scan,
                    "raw_key_profile": monthly_raw_metrics,
                    "post_exact_collapse_canonical_key_profile": monthly_post_metrics,
                    "identity": monthly_identity.public(),
                    "canonicalization": (
                        "global exact four-part-key collapse then unique inclusive "
                        "MthPrcDt/MthCalDt interval selection"
                    ),
                },
                "privacy": {
                    "raw_security_keys_returned": False,
                    "canonical_rows_returned": False,
                    "raw_row_examples_returned": False,
                    "durable_files_written": False,
                    "quality_artifact_or_pointer_written": False,
                    "scratch_destroyed_before_return": True,
                    "output_is_aggregate_canonical_json_only": True,
                },
                "downstream": {
                    "typed_backbone_authorized_by_this_discovery": False,
                    "provenance_authorization_granted": False,
                    "empirical_lock_promotion_authorized": False,
                    "production_publication_authorized": False,
                    "return_history_authorized": False,
                    "beta_estimation_authorized": False,
                    "labels_authorized": False,
                    "backtests_authorized": False,
                },
            }
            # Serialization is part of the in-memory contract.  This also
            # proves no non-JSON private key object escaped into the result.
            canonical_json_bytes(result)
        # TemporaryDirectory has destroyed all packed raw keys at this point.
        for source in opened.values():
            source.assert_stable()
        return result


run_typed_backbone_discovery = discover_typed_backbone_identity_joins
