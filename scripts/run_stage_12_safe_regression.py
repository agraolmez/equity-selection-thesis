#!/usr/bin/env python3
"""Run the frozen Stage 12 synthetic tests behind a fail-closed I/O guard.

This command is intentionally narrower than a repository-wide pytest run.  A
repository-wide run contains tests that open production raw/interim/processed
payloads, which is incompatible with the Stage 12 metadata-only authority.
Only the explicitly reviewed synthetic protocol tests below may be selected.
"""

from __future__ import annotations

import builtins
import functools
import hashlib
import json
import os
import re
import site
import sys
import urllib.parse
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = 1
PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIT_OPTION = "--stage12-access-audit="

# A whole file is permitted only when every test in it is synthetic and
# metadata-only.  The trial-ledger file has one production-data test, so its
# two safe cases must be named exactly.
SAFE_WHOLE_TEST_FILES = frozenset(
    {
        "tests/unit/test_stage12_final_freeze.py",
        "tests/unit/test_final_test_calendar.py",
        "tests/unit/test_final_test_authorization.py",
        "tests/unit/test_final_test_access.py",
        "tests/unit/test_final_input_construction.py",
        "tests/unit/test_final_input_materialization.py",
        "tests/unit/test_final_raw_source_publication.py",
        "tests/unit/test_final_source_builder.py",
        "tests/unit/test_final_session_authority.py",
        "tests/unit/test_final_terminal_event_overlay.py",
        "tests/unit/test_final_terminal_event_sensitivity.py",
        "tests/unit/test_final_test_checkpoint.py",
        "tests/unit/test_final_test_benchmarks.py",
        "tests/unit/test_final_test_economic_kernel.py",
        "tests/unit/test_final_test_production_runtime.py",
        "tests/unit/test_final_test_production_results.py",
        "tests/unit/test_final_test_results.py",
        "tests/unit/test_stage14_inference.py",
        "tests/unit/test_numerical_materiality.py",
        "tests/unit/test_stage12_numerical_materiality_amendment.py",
        "tests/unit/test_stage11_validation_empirical_feature_stream.py",
        "tests/unit/test_stage11_empirical_execution.py",
        "tests/unit/test_stage9_selector_preprocessing.py",
        "tests/unit/test_stage13_source_checkpoint_context_migration.py",
        "tests/unit/test_stage13_decision_checkpoint_rebind.py",
        "tests/unit/test_final_test_orchestration.py",
        "tests/unit/test_final_test_publication.py",
    }
)
SAFE_EXACT_NODE_IDS = frozenset(
    {
        "tests/unit/test_stage12_trial_ledger.py::"
        "test_synthetic_ledger_is_complete_deterministic_and_nonduplicative",
        "tests/unit/test_stage12_trial_ledger.py::"
        "test_trial_source_hash_and_configuration_identity_fail_closed",
        "tests/unit/test_stage11_validation_market_extension.py::"
        "test_entry_projection_schema_is_exact_entry_consumer_only",
        "tests/unit/test_stage11_validation_market_extension.py::"
        "test_entry_consumer_transform_matches_frozen_stage2_without_return_fields",
    }
)


class GuardViolation(PermissionError):
    """Raised before a forbidden payload can be opened or a child can escape."""


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def _selected_test_arguments(arguments: Iterable[str]) -> tuple[str, ...]:
    selected: list[str] = []
    for value in arguments:
        if value.startswith("-"):
            continue
        if value.endswith((".xml", ".json")) and "=" not in value:
            # Values belonging to pytest output options are never test selectors.
            continue
        if value.startswith("tests/"):
            selected.append(value)
    if not selected:
        raise SystemExit("safe regression requires at least one explicit test selector")
    for node_id in selected:
        file_name = node_id.split("::", 1)[0]
        if node_id in SAFE_EXACT_NODE_IDS:
            continue
        if file_name in SAFE_WHOLE_TEST_FILES and "::" not in node_id:
            continue
        raise SystemExit(
            f"test selector is outside the frozen safe allowlist: {node_id}"
        )
    return tuple(selected)


def _load_local_json(path: Path) -> tuple[Mapping[str, Any], bytes]:
    payload = path.read_bytes()
    value = json.loads(payload)
    if not isinstance(value, Mapping):
        raise SystemExit(f"Stage 12 guard binding is not a JSON object: {path}")
    return value, payload


def _stage11_manifest_binding() -> tuple[Path, str]:
    """Resolve one hash-pinned Stage 11 metadata manifest before guarding."""

    policy, _ = _load_local_json(
        PROJECT_ROOT / "configs/phase_06_final_test/stage_12_freeze_policy_v1.json"
    )
    amendment, _ = _load_local_json(
        PROJECT_ROOT
        / "configs/phase_06_final_test/stage_12_materiality_amendment_v1.json"
    )
    pointer_path = (
        PROJECT_ROOT
        / "artifacts/phase_05_validation/LATEST_STAGE_11_EXECUTION_VERIFIED.json"
    )
    pointer, pointer_payload = _load_local_json(pointer_path)
    authorities = policy.get("required_authorities")
    if not isinstance(authorities, list):
        raise SystemExit("Stage 12 authority registry is missing")
    expected_pointer_sha = next(
        (
            str(item.get("sha256"))
            for item in authorities
            if isinstance(item, Mapping) and item.get("role") == "validation_selection"
        ),
        "",
    )
    if hashlib.sha256(pointer_payload).hexdigest() != expected_pointer_sha:
        raise SystemExit("Stage 11 verified pointer differs from Stage 12 policy")
    authority = amendment.get("authority_boundary")
    if not isinstance(authority, Mapping):
        raise SystemExit("Stage 11 amendment authority boundary is missing")
    build_id = str(authority.get("stage11_primary_build_id"))
    manifest_sha = str(authority.get("stage11_primary_manifest_sha256"))
    if (
        pointer.get("primary_build_id") != build_id
        or pointer.get("primary_manifest_sha256") != manifest_sha
    ):
        raise SystemExit("Stage 11 manifest binding differs from amendment")
    bundle = Path(str(pointer.get("primary_bundle_path"))).absolute()
    permitted_parent = (
        Path.home()
        / "Desktop/Thesis/interim/phase_05_validation/stage_11_execution_v1/"
        "execution_results/primary"
    ).absolute()
    expected_bundle = permitted_parent / build_id
    if bundle != expected_bundle or ".." in bundle.parts:
        raise SystemExit(
            "Stage 11 metadata bundle path is outside its frozen namespace"
        )
    return bundle / "manifest.json", manifest_sha


class AccessGuard:
    """Layered Python/native-reader guard with an auditable zero-open receipt."""

    def __init__(
        self,
        project_root: Path,
        *,
        authorized_external_metadata: Mapping[Path, str] | None = None,
    ) -> None:
        self.project_root = project_root.resolve(strict=True)
        self.forbidden_roots = tuple(
            path.resolve(strict=False)
            for path in (
                self.project_root / "raw",
                self.project_root / "interim",
                self.project_root / "processed",
                Path.home() / "Desktop",
            )
        )
        self.audit_events_seen = 0
        self.filesystem_open_events_seen = 0
        self.forbidden_access_attempts_denied = 0
        self.process_spawn_attempts_denied = 0
        self.native_reader_calls_checked = 0
        self.native_path_values_checked = 0
        self.duckdb_sql_entrypoint_calls_checked = 0
        self.authenticated_external_metadata_open_events = 0
        self.authorized_external_metadata = {
            path.absolute().resolve(strict=False): str(digest)
            for path, digest in (authorized_external_metadata or {}).items()
        }
        self._patched: set[tuple[int, str]] = set()
        self.dependency_site_path: str | None = None

    @staticmethod
    def _coerce_path(value: object) -> Path | None:
        if isinstance(value, int):
            return None
        if isinstance(value, bytes):
            try:
                value = os.fsdecode(value)
            except UnicodeDecodeError:
                return None
        if not isinstance(value, (str, os.PathLike)):
            return None
        try:
            return Path(value).expanduser().absolute().resolve(strict=False)
        except (OSError, RuntimeError, ValueError):
            return None

    def _forbidden_root(self, value: object) -> Path | None:
        selected = self._coerce_path(value)
        if selected is None:
            return None
        for root in self.forbidden_roots:
            try:
                selected.relative_to(root)
            except ValueError:
                continue
            return root
        return None

    def _is_authorized_external_metadata(self, value: object) -> bool:
        selected = self._coerce_path(value)
        return selected is not None and selected in self.authorized_external_metadata

    @staticmethod
    def _write_requested(args: tuple[object, ...]) -> bool:
        if len(args) >= 2 and isinstance(args[1], str):
            return any(flag in args[1] for flag in ("w", "a", "x", "+"))
        if len(args) >= 3 and isinstance(args[2], int):
            write_flags = (
                os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
            )
            return bool(args[2] & write_flags)
        return False

    def check_path(self, value: object, *, operation: str) -> None:
        self.native_reader_calls_checked += 1
        self.native_path_values_checked += 1
        root = self._forbidden_root(value)
        if root is None:
            return
        self.forbidden_access_attempts_denied += 1
        raise GuardViolation(
            f"Stage 12 safe regression denied {operation} below {root}"
        )

    def audit_hook(self, event: str, args: tuple[object, ...]) -> None:
        self.audit_events_seen += 1
        if event == "open":
            self.filesystem_open_events_seen += 1
            if args:
                if self._is_authorized_external_metadata(args[0]):
                    if self._write_requested(args):
                        self.forbidden_access_attempts_denied += 1
                        raise GuardViolation(
                            "Stage 12 authenticated external metadata is read-only"
                        )
                    self.authenticated_external_metadata_open_events += 1
                    return
                root = self._forbidden_root(args[0])
                if root is not None:
                    self.forbidden_access_attempts_denied += 1
                    raise GuardViolation(
                        f"Stage 12 safe regression denied filesystem open below {root}"
                    )
            return
        if event in {
            "subprocess.Popen",
            "os.system",
            "os.exec",
            "os.posix_spawn",
            "os.spawn",
            "pty.spawn",
        }:
            self.process_spawn_attempts_denied += 1
            raise GuardViolation(
                "Stage 12 safe regression forbids child-process escape"
            )

    def _check_nested_paths(self, value: object, *, operation: str) -> None:
        if isinstance(value, Mapping):
            for nested in value.values():
                self._check_nested_paths(nested, operation=operation)
            return
        if isinstance(value, (list, tuple, set, frozenset)):
            for nested in value:
                self._check_nested_paths(nested, operation=operation)
            return
        self.check_path(value, operation=operation)

    @staticmethod
    def _owner_label(owner: object) -> str:
        module = getattr(owner, "__module__", None)
        name = getattr(owner, "__name__", None)
        if isinstance(module, str) and isinstance(name, str):
            return f"{module}.{name}"
        if isinstance(owner, ModuleType):
            return owner.__name__
        return type(owner).__name__

    def _patch_path_function(
        self,
        owner: object,
        name: str,
        *,
        path_positions: tuple[int, ...] = (0, 1),
    ) -> None:
        key = (id(owner), name)
        if key in self._patched or not hasattr(owner, name):
            return
        original = getattr(owner, name)
        if not callable(original):
            return
        operation = f"{self._owner_label(owner)}.{name}"

        @functools.wraps(original)
        def guarded(*args: object, **kwargs: object) -> Any:
            for position in path_positions:
                if position < len(args):
                    self._check_nested_paths(args[position], operation=operation)
            for keyword in (
                "source",
                "where",
                "path",
                "path_or_paths",
                "path_or_buffer",
                "filename",
                "file_glob",
                "file_globs",
            ):
                if keyword in kwargs:
                    self._check_nested_paths(kwargs[keyword], operation=operation)
            return original(*args, **kwargs)

        try:
            setattr(owner, name, guarded)
        except (AttributeError, TypeError) as exc:
            raise SystemExit(
                f"required native-reader guard is not installable: {operation}"
            ) from exc
        self._patched.add(key)

    @staticmethod
    def _sql_path_literal(value: str) -> object:
        decoded = urllib.parse.unquote(value.strip())
        if decoded.startswith("file://"):
            parsed = urllib.parse.urlparse(decoded)
            return urllib.parse.unquote(parsed.path)
        return decoded

    @staticmethod
    def _looks_path_like_sql_literal(value: str) -> bool:
        decoded = urllib.parse.unquote(value.strip())
        return (
            decoded.startswith(("/", "./", "../", "~", "file://"))
            or "/" in decoded
            or "\\" in decoded
            or any(mark in decoded for mark in ("*", "?", "[", "]"))
        )

    def _check_duckdb_query(self, value: object, *, operation: str) -> None:
        if not isinstance(value, str):
            return
        self.duckdb_sql_entrypoint_calls_checked += 1
        decoded = urllib.parse.unquote(value)
        for root in self.forbidden_roots:
            if str(root) in decoded or root.as_posix() in decoded:
                self.forbidden_access_attempts_denied += 1
                raise GuardViolation(
                    f"Stage 12 safe regression denied {operation} SQL below {root}"
                )
        for match in re.finditer(r"'((?:''|[^'])*)'", decoded):
            literal = match.group(1).replace("''", "'")
            if self._looks_path_like_sql_literal(literal):
                self._check_nested_paths(
                    self._sql_path_literal(literal),
                    operation=operation,
                )

    def _patch_duckdb_sql_function(
        self,
        owner: object,
        name: str,
        *,
        query_position: int,
        parameters_position: int | None,
    ) -> None:
        key = (id(owner), name)
        if key in self._patched or not hasattr(owner, name):
            return
        original = getattr(owner, name)
        if not callable(original):
            return
        operation = f"{self._owner_label(owner)}.{name}"

        @functools.wraps(original)
        def guarded(*args: object, **kwargs: object) -> Any:
            if query_position < len(args):
                self._check_duckdb_query(args[query_position], operation=operation)
            for keyword in ("query", "query_string", "sql"):
                if keyword in kwargs:
                    self._check_duckdb_query(kwargs[keyword], operation=operation)
            if parameters_position is not None and parameters_position < len(args):
                self._check_nested_paths(
                    args[parameters_position],
                    operation=f"{operation} parameters",
                )
            for keyword in ("parameters", "params"):
                if keyword in kwargs:
                    self._check_nested_paths(
                        kwargs[keyword],
                        operation=f"{operation} parameters",
                    )
            return original(*args, **kwargs)

        try:
            setattr(owner, name, guarded)
        except (AttributeError, TypeError) as exc:
            raise SystemExit(
                f"required DuckDB SQL guard is not installable: {operation}"
            ) from exc
        self._patched.add(key)

    def patch_loaded_native_readers(self) -> None:
        """Patch native readers whose C/C++ opens are not Python audit events."""

        module_functions = {
            "pyarrow": ("memory_map", "OSFile", "input_stream", "output_stream"),
            "pyarrow.parquet": (
                "ParquetDataset",
                "read_table",
                "read_metadata",
                "read_schema",
                "write_table",
            ),
            "pyarrow.csv": ("read_csv", "open_csv", "write_csv"),
            "pyarrow.dataset": ("dataset", "write_dataset"),
            "pandas": (
                "read_csv",
                "read_parquet",
                "read_feather",
                "read_pickle",
                "read_json",
                "read_excel",
                "read_hdf",
            ),
            "polars": ("read_csv", "read_parquet", "scan_csv", "scan_parquet"),
        }
        for module_name, names in module_functions.items():
            module = sys.modules.get(module_name)
            if not isinstance(module, ModuleType):
                continue
            for name in names:
                positions = (0,)
                if module_name == "pyarrow.parquet" and name == "write_table":
                    positions = (1,)
                if module_name == "pyarrow.dataset" and name == "write_dataset":
                    positions = (1,)
                self._patch_path_function(module, name, path_positions=positions)

            if module_name == "pyarrow.parquet":
                parquet_file = getattr(module, "ParquetFile", None)
                if parquet_file is not None:
                    self._patch_path_function(
                        parquet_file,
                        "__init__",
                        path_positions=(1,),
                    )

        duckdb = sys.modules.get("duckdb")
        if isinstance(duckdb, ModuleType):
            self._patch_path_function(duckdb, "connect", path_positions=(0,))
            connection_type = getattr(duckdb, "DuckDBPyConnection", None)
            if connection_type is None:
                return
            for owner, query_position, parameters_position in (
                (duckdb, 0, 1),
                (connection_type, 1, 2),
            ):
                for name in ("execute", "executemany", "sql", "query", "from_query"):
                    self._patch_duckdb_sql_function(
                        owner,
                        name,
                        query_position=query_position,
                        parameters_position=parameters_position,
                    )
            for owner, path_position in ((duckdb, 0), (connection_type, 1)):
                for name in (
                    "read_parquet",
                    "from_parquet",
                    "read_csv",
                    "from_csv_auto",
                    "read_json",
                ):
                    self._patch_path_function(
                        owner,
                        name,
                        path_positions=(path_position,),
                    )

    def verify_loaded_native_reader_guards(self) -> None:
        """Fail if a loaded required extension escaped its in-place wrapper."""

        parquet = sys.modules.get("pyarrow.parquet")
        if isinstance(parquet, ModuleType):
            parquet_file = getattr(parquet, "ParquetFile", None)
            if (
                parquet_file is not None
                and (
                    id(parquet_file),
                    "__init__",
                )
                not in self._patched
            ):
                raise SystemExit("required PyArrow ParquetFile guard is not active")
        duckdb = sys.modules.get("duckdb")
        if isinstance(duckdb, ModuleType):
            connection_type = getattr(duckdb, "DuckDBPyConnection", None)
            if connection_type is not None and any(
                key not in self._patched
                for key in (
                    (id(duckdb), "connect"),
                    (id(connection_type), "execute"),
                    (id(connection_type), "read_parquet"),
                )
            ):
                raise SystemExit("required DuckDB native guards are not active")

    def install_import_watcher(self) -> None:
        original_import = builtins.__import__

        @functools.wraps(original_import)
        def guarded_import(*args: object, **kwargs: object) -> Any:
            imported = original_import(*args, **kwargs)
            self.patch_loaded_native_readers()
            return imported

        builtins.__import__ = guarded_import

    def enable_audited_dependency_site(self) -> None:
        """Add the canonical user dependency directory only after guard install.

        The local environment stores several locked direct dependencies there.
        ``PYTHONNOUSERSITE=1`` prevents startup-time code execution from that
        directory; adding it now makes all subsequent imports observable.
        """

        selected = Path(site.getusersitepackages()).absolute().resolve(strict=False)
        expected = (
            (
                Path.home()
                / f".local/lib/python{sys.version_info.major}.{sys.version_info.minor}/"
                "site-packages"
            )
            .absolute()
            .resolve(strict=False)
        )
        if selected != expected or not selected.is_dir():
            raise SystemExit("locked user dependency site path differs")
        if str(selected) not in sys.path:
            sys.path.append(str(selected))
        self.dependency_site_path = str(selected)

    def authenticate_stage11_metadata_closure(
        self,
        *,
        manifest_path: Path,
        manifest_sha256: str,
    ) -> None:
        """Authenticate the manifest and admit only its one JSON lineage leaf."""

        payload = manifest_path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != manifest_sha256:
            raise SystemExit("Stage 11 external metadata manifest SHA-256 differs")
        manifest = json.loads(payload)
        if not isinstance(manifest, Mapping):
            raise SystemExit("Stage 11 external metadata manifest is not an object")
        hashes = manifest.get("implementation_code_hashes")
        if not isinstance(hashes, Mapping):
            raise SystemExit("Stage 11 implementation inventory is missing")
        external_root = (Path.home() / "Desktop/Thesis").absolute()
        clean_root = manifest_path.parent.parents[1] / "clean_portfolio_candidates"
        for name, digest in hashes.items():
            text = str(name)
            if not text.startswith("private_clean_portfolio_candidate/"):
                continue
            relative = Path(text).relative_to("private_clean_portfolio_candidate")
            candidate = (clean_root / relative).absolute().resolve(strict=False)
            try:
                candidate.relative_to(external_root)
            except ValueError as exc:
                raise SystemExit(
                    "Stage 11 clean-candidate metadata escaped its frozen root"
                ) from exc
            if (
                ".." in relative.parts
                or candidate.suffix != ".json"
                or not isinstance(digest, str)
                or len(digest) != 64
            ):
                raise SystemExit("Stage 11 clean-candidate metadata binding is unsafe")
            self.authorized_external_metadata[candidate] = digest

    def receipt(
        self, *, selected_tests: tuple[str, ...], exit_code: int
    ) -> dict[str, Any]:
        selected_sha = hashlib.sha256(
            _canonical_json_bytes(list(selected_tests))
        ).hexdigest()
        clean = (
            self.forbidden_access_attempts_denied == 0
            and self.process_spawn_attempts_denied == 0
        )
        return {
            "schema_version": SCHEMA_VERSION,
            "status": (
                "NO_FORBIDDEN_PAYLOAD_ACCESS" if clean else "DENIED_ATTEMPT_OBSERVED"
            ),
            "guard_scope": {
                "forbidden_roots": [str(path) for path in self.forbidden_roots],
                "child_processes_permitted": False,
                "native_reader_wrappers_installed": True,
                "explicit_test_allowlist_required": True,
                "audited_dependency_site_path": self.dependency_site_path,
                "authorized_external_metadata": {
                    str(path): digest
                    for path, digest in sorted(
                        self.authorized_external_metadata.items(),
                        key=lambda item: str(item[0]),
                    )
                },
                "measurement_scope": (
                    "this safe-regression subprocess, from AccessGuard audit-hook "
                    "installation through pytest.main return"
                ),
                "zero_open_claim": (
                    "No forbidden-root dispatch succeeded through the enumerated "
                    "Python-audit, native-reader, and DuckDB entry points during "
                    "this guarded test execution."
                ),
                "excluded_scope": [
                    "activity before guard installation",
                    "the parent Stage 12 build process",
                    "prior amendment or Stage 13 executions",
                    "external processes",
                    "unenumerated native entry points",
                ],
                "guarded_native_entrypoints": [
                    "Python audit open/process-spawn events",
                    "PyArrow ParquetFile and parquet/csv/dataset path APIs",
                    "Pandas and Polars reader path APIs",
                    "DuckDB connect, SQL/query, and direct reader path APIs",
                ],
            },
            "selected_tests": list(selected_tests),
            "selected_tests_sha256": selected_sha,
            "pytest_exit_code": exit_code,
            "counters": {
                "audit_events_seen": self.audit_events_seen,
                "filesystem_open_events_seen": self.filesystem_open_events_seen,
                "native_reader_calls_checked": self.native_reader_calls_checked,
                "native_path_values_checked": self.native_path_values_checked,
                "duckdb_sql_entrypoint_calls_checked": (
                    self.duckdb_sql_entrypoint_calls_checked
                ),
                "authenticated_external_metadata_open_events": (
                    self.authenticated_external_metadata_open_events
                ),
                "forbidden_access_attempts_denied": self.forbidden_access_attempts_denied,
                "process_spawn_attempts_denied": self.process_spawn_attempts_denied,
                "raw_payload_files_opened": 0,
                "interim_payload_files_opened": 0,
                "processed_payload_files_opened": 0,
                "desktop_payload_files_opened": 0,
                "final_period_payload_files_opened": 0,
                "final_period_rows_read": 0,
            },
        }


def _extract_audit_path(arguments: list[str]) -> tuple[Path, list[str]]:
    matches = [value for value in arguments if value.startswith(AUDIT_OPTION)]
    if len(matches) != 1:
        raise SystemExit("exactly one --stage12-access-audit path is required")
    value = matches[0][len(AUDIT_OPTION) :]
    selected = Path(value)
    if not selected.is_absolute() or ".." in selected.parts:
        raise SystemExit("Stage 12 access-audit path must be absolute and bounded")
    return selected, [value for value in arguments if value not in matches]


def main() -> int:
    audit_path, pytest_arguments = _extract_audit_path(sys.argv[1:])
    selected_tests = _selected_test_arguments(pytest_arguments)
    stage11_manifest, stage11_manifest_sha = _stage11_manifest_binding()
    guard = AccessGuard(
        PROJECT_ROOT,
        authorized_external_metadata={stage11_manifest: stage11_manifest_sha},
    )
    sys.addaudithook(guard.audit_hook)
    guard.authenticate_stage11_metadata_closure(
        manifest_path=stage11_manifest,
        manifest_sha256=stage11_manifest_sha,
    )
    guard.install_import_watcher()
    guard.enable_audited_dependency_site()
    guard.patch_loaded_native_readers()

    import pytest

    exit_code = int(pytest.main(pytest_arguments))
    guard.patch_loaded_native_readers()
    guard.verify_loaded_native_reader_guards()
    receipt = guard.receipt(selected_tests=selected_tests, exit_code=exit_code)
    audit_path.write_bytes(_canonical_json_bytes(receipt))
    if receipt["status"] != "NO_FORBIDDEN_PAYLOAD_ACCESS":
        return 86
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
