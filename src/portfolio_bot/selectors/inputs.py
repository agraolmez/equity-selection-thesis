"""Authenticate and deliver the closed development-only Stage 9 inputs."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from portfolio_bot.selectors.contract import (
    DEFAULT_POLICY,
    REQUIRED_INPUT_NAMES,
    SelectorContractError,
    project_path,
    require,
    sha256_file,
    validate_contract,
)


FIRST_SIGNAL = dt.date(2000, 1, 31)
FIRST_LONG_SIGNAL = dt.date(2000, 6, 30)
LAST_SIGNAL = dt.date(2014, 10, 31)
REQUIRED_LONG_SIGNAL_DATES = 173

FEATURE_SPINE_COLUMNS = (
    "signal_date",
    "permno",
    "permco",
    "long_eligible",
    "short_eligible",
)
LABEL_COLUMNS = (
    "signal_date",
    "permno",
    "permco",
    "short_eligible",
    "holding_month",
    "entry_date",
    "exit_date",
    "security_label_available_date",
    "month_label_available_date",
    "unranked_excess_return",
    "unranked_residual_return",
    "residual_rank_numerator",
    "excess_rank_numerator",
    "residual_return_rank",
    "excess_return_rank",
    "residual_return_mad_standardized",
    "outcome_status",
    "primary_label_status",
    "magnitude_label_status",
)
SPLIT_COLUMNS = (
    "fold_id",
    "signal_date",
    "holding_month",
    "fit_signal_date",
    "role",
    "month_label_available_date",
    "available_by_fit",
    "included_in_training",
    "included_in_validation",
    "purged_or_embargoed",
)


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


def _schema_sha256(schema: pa.Schema) -> str:
    return hashlib.sha256(schema.serialize().to_pybytes()).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text("utf-8"))
    require(isinstance(value, dict), f"Expected Stage 9 JSON object: {path}")
    return value


def _authenticate_project_file(path_value: object, expected: object, label: str) -> Path:
    path = project_path(str(path_value))
    require(path.is_file() and not path.is_symlink(), f"Unsafe Stage 9 {label} path")
    require(sha256_file(path) == str(expected), f"Stage 9 {label} hash differs")
    return path


def _anchored_regular_path(
    candidate: Path,
    *,
    trusted_root: Path,
    label: str,
) -> Path:
    """Reject lexical escapes and every symlinked path component."""

    root = Path(os.path.abspath(trusted_root.expanduser()))
    raw = Path(os.path.abspath(candidate.expanduser()))
    try:
        relative = raw.relative_to(root)
    except ValueError as exc:
        raise SelectorContractError(f"{label} escaped the Stage 9 database root") from exc
    require(not root.is_symlink(), "Symlinked Stage 9 database root is prohibited")
    current = root
    for part in relative.parts:
        current = current / part
        require(not current.is_symlink(), f"Symlinked Stage 9 input is prohibited: {current}")
    try:
        resolved = raw.resolve(strict=True)
        resolved.relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise SelectorContractError(f"Unsafe or absent Stage 9 input: {raw}") from exc
    require(resolved.is_file(), f"Stage 9 input is not a regular file: {resolved}")
    return resolved


@dataclass(frozen=True)
class SelectorInputTable:
    gate: str
    name: str
    _path: Path
    sha256: str
    schema_sha256: str
    size_bytes: int
    rows: int
    row_groups: int
    predictor_permitted: bool
    filter_permitted: bool
    key_audit_only: bool
    role: str
    stat_identity: tuple[int, ...]

    def receipt(self) -> dict[str, Any]:
        return {
            "gate": self.gate,
            "name": self.name,
            "sha256": self.sha256,
            "schema_sha256": self.schema_sha256,
            "size_bytes": self.size_bytes,
            "rows": self.rows,
            "row_groups": self.row_groups,
            "predictor_permitted": self.predictor_permitted,
            "filter_permitted": self.filter_permitted,
            "key_audit_only": self.key_audit_only,
            "role": self.role,
        }


@dataclass(frozen=True)
class Stage9Inputs:
    """Authenticated catalog with purpose-bounded read methods.

    There is intentionally no generic path accessor.  In particular, the
    Stage 8 numerical payload can only be projected to its two primary-key
    fields by :meth:`read_stage8_key_audit`.
    """

    database_root: Path
    authority: Mapping[str, Any]
    tables: Mapping[str, SelectorInputTable]
    input_receipts: tuple[Mapping[str, Any], ...]
    hashes_authenticated: bool = True

    def _trusted_path(self, name: str) -> Path:
        require(self.hashes_authenticated, "Unauthenticated Stage 9 catalog cannot deliver data")
        try:
            table = self.tables[name]
        except KeyError as exc:
            raise SelectorContractError(f"Unregistered Stage 9 input: {name}") from exc
        path = _anchored_regular_path(
            table._path,
            trusted_root=self.database_root,
            label=f"held Stage 9 input {name}",
        )
        require(
            _stat_identity(path.stat()) == table.stat_identity,
            f"Stage 9 input identity changed: {name}",
        )
        return path

    def read_feature_dictionary(self) -> pa.Table:
        table = pq.read_table(
            self._trusted_path("feature_dictionary.parquet"),
            use_threads=False,
        ).combine_chunks()
        require(table.num_rows == 163, "Stage 9 feature dictionary row count differs")
        rows = table.to_pylist()
        require(
            [int(row["feature_ordinal"]) for row in rows] == list(range(1, 164)),
            "Stage 9 feature dictionary is not in frozen ordinal order",
        )
        kinds = [str(row["kind"]) for row in rows]
        require(
            kinds.count("continuous_firm") == 122
            and kinds.count("binary_firm") == 8
            and kinds.count("aggregate_state") == 23
            and kinds.count("categorical_firm") == 8
            and kinds.count("categorical_multihot_firm") == 2,
            "Stage 9 feature-kind counts differ",
        )
        return table

    def read_feature_set_membership(self) -> pa.Table:
        table = pq.read_table(
            self._trusted_path("feature_set_membership.parquet"),
            use_threads=False,
        ).combine_chunks()
        require(table.num_rows == 489, "Stage 9 feature-set membership row count differs")
        rows = table.to_pylist()
        full = [
            row for row in rows
            if row["feature_set"] == "full_feature_primary" and bool(row["included"])
        ]
        require(len(full) == 163, "Stage 9 full-feature primary membership differs")
        return table

    def read_category_vocabulary(self) -> pa.Table:
        table = pq.read_table(
            self._trusted_path("category_vocabulary.parquet"),
            use_threads=False,
        ).combine_chunks()
        require(table.num_rows == 245, "Stage 9 category vocabulary width differs")
        rows = table.to_pylist()
        encoded = [str(row["encoded_column_name"]) for row in rows]
        require(len(encoded) == len(set(encoded)), "Stage 9 encoded category column is duplicated")
        previous: dict[str, int] = {}
        for row in rows:
            feature = str(row["feature_name"])
            ordinal = int(row["category_ordinal"])
            require(
                ordinal == previous.get(feature, 0) + 1,
                f"Stage 9 category order differs for {feature}",
            )
            previous[feature] = ordinal
        return table

    def feature_names(self) -> tuple[str, ...]:
        return tuple(str(value) for value in self.read_feature_dictionary()["feature_name"].to_pylist())

    def read_development_feature_panel(
        self,
        columns: Sequence[str] | None = None,
    ) -> pa.Table:
        """Read only long-eligible 2000--2014 rows from the Stage 6 panel."""

        feature_names = self.feature_names()
        permitted = set(FEATURE_SPINE_COLUMNS) | set(feature_names)
        requested = (
            (*FEATURE_SPINE_COLUMNS, *feature_names)
            if columns is None
            else tuple(str(name) for name in columns)
        )
        require(len(requested) == len(set(requested)), "Duplicate Stage 9 feature projection")
        require(set(requested).issubset(permitted), "Stage 9 feature projection is not frozen")
        guards = ("signal_date", "permno", "long_eligible")
        projection = tuple(dict.fromkeys((*requested, *guards)))
        raw = pq.read_table(
            self._trusted_path("feature_store_raw.parquet"),
            columns=list(projection),
            use_threads=False,
        )
        bounded = raw.filter(pc.fill_null(raw["long_eligible"], False)).select(list(requested))
        require(bounded.num_rows == 286_628, "Stage 9 long-eligible feature rows differ")
        _require_development_key_table(bounded, label="Stage 9 feature panel")
        return bounded

    def read_aggregate_signal_history(self) -> pa.Table:
        """Return one authenticated aggregate-state row for all 178 signals.

        The first five signal dates have no long-eligible securities, but they
        remain legitimate strictly-prior months for later expanding aggregate
        z-scores.  This reader exposes only the date and the 23 registered
        aggregate-state predictors.
        """

        dictionary = self.read_feature_dictionary().to_pandas()
        names = dictionary.loc[
            dictionary["kind"].eq("aggregate_state"), "feature_name"
        ].astype(str).tolist()
        require(len(names) == 23, "Stage 9 aggregate-state registry differs")
        table = pq.read_table(
            self._trusted_path("feature_store_raw.parquet"),
            columns=["signal_date", *names],
            use_threads=False,
        ).combine_chunks()
        require(table.num_rows == 1_263_048, "Stage 9 aggregate audit spine differs")
        frame = table.to_pandas()
        grouped = frame.groupby("signal_date", sort=True, observed=True)
        for name in names:
            variation = grouped[name].nunique(dropna=False)
            require(
                bool((variation <= 1).all()),
                f"Aggregate-state value varies within signal date: {name}",
            )
        collapsed = grouped[names].first().reset_index()
        require(
            len(collapsed) == 178
            and collapsed["signal_date"].iloc[0] == FIRST_SIGNAL
            and collapsed["signal_date"].iloc[-1] == LAST_SIGNAL,
            "Stage 9 aggregate signal history differs",
        )
        fields = [pa.field("signal_date", pa.date32(), nullable=False)]
        fields.extend(pa.field(name, pa.float64()) for name in names)
        return pa.Table.from_pandas(
            collapsed,
            schema=pa.schema(fields),
            preserve_index=False,
            safe=True,
        )

    def read_development_labels(self) -> pa.Table:
        table = pq.read_table(
            self._trusted_path("development_labels.parquet"),
            columns=list(LABEL_COLUMNS),
            use_threads=False,
        ).combine_chunks()
        require(table.num_rows == 286_628, "Stage 9 development label rows differ")
        _require_development_key_table(table, label="Stage 9 labels")
        present = pc.sum(pc.equal(table["primary_label_status"], "PRESENT")).as_py()
        require(int(present or 0) == 286_322, "Stage 9 PRESENT target count differs")
        return table

    def read_split_membership(self) -> pa.Table:
        table = pq.read_table(
            self._trusted_path("label_split_membership.parquet"),
            columns=list(SPLIT_COLUMNS),
            use_threads=False,
        ).combine_chunks()
        require(table.num_rows == 712, "Stage 9 split-membership row count differs")
        folds = set(table["fold_id"].to_pylist())
        require(
            folds == {"dev_fold_1", "dev_fold_2", "dev_fold_3", "dev_fold_4"},
            "Stage 9 development fold domain differs",
        )
        require(
            pc.min(table["signal_date"]).as_py() == FIRST_SIGNAL
            and pc.max(table["signal_date"]).as_py() == LAST_SIGNAL,
            "Stage 9 split membership escaped development scope",
        )
        return table

    def read_stage8_key_audit(self) -> pa.Table:
        """Read only Stage 8 primary keys; no numerical field can escape."""

        specification = self.tables["development_risk_cost_inputs.parquet"]
        require(
            specification.key_audit_only
            and not specification.predictor_permitted
            and not specification.filter_permitted,
            "Stage 8 payload is not key-audit-only",
        )
        table = pq.read_table(
            self._trusted_path(specification.name),
            columns=["signal_date", "permno"],
            use_threads=False,
        ).combine_chunks()
        require(table.num_rows == 286_628, "Stage 8 key-audit row count differs")
        _require_development_key_table(table, label="Stage 8 key audit")
        return table

    def reconcile_key_domains(self) -> dict[str, int]:
        feature = self.read_development_feature_panel(("signal_date", "permno"))
        labels = self.read_development_labels().select(["signal_date", "permno"])
        risk = self.read_stage8_key_audit()
        require(_same_keys(feature, labels), "Stage 6 feature and Stage 7 label keys differ")
        require(_same_keys(feature, risk), "Stage 8 key audit differs from selector keys")
        return {
            "feature_rows": feature.num_rows,
            "label_rows": labels.num_rows,
            "stage8_key_rows": risk.num_rows,
            "key_differences": 0,
        }


def _require_development_key_table(table: pa.Table, *, label: str) -> None:
    require(
        "signal_date" in table.schema.names and "permno" in table.schema.names,
        f"{label} omitted its primary key",
    )
    require(
        table["signal_date"].null_count == 0 and table["permno"].null_count == 0,
        f"{label} has null primary keys",
    )
    require(
        pc.min(table["signal_date"]).as_py() == FIRST_LONG_SIGNAL
        and pc.max(table["signal_date"]).as_py() == LAST_SIGNAL,
        f"{label} escaped the Stage 9 development window",
    )
    require(
        int(pc.count_distinct(table["signal_date"]).as_py()) == REQUIRED_LONG_SIGNAL_DATES,
        f"{label} signal-date domain differs",
    )
    days = pc.cast(table["signal_date"], pa.int32()).to_numpy(zero_copy_only=False)
    permnos = table["permno"].to_numpy(zero_copy_only=False)
    if table.num_rows > 1:
        violation = (days[1:] < days[:-1]) | (
            (days[1:] == days[:-1]) & (permnos[1:] <= permnos[:-1])
        )
        require(not bool(np.any(violation)), f"{label} key is not strict sorted")


def _same_keys(left: pa.Table, right: pa.Table) -> bool:
    if left.num_rows != right.num_rows:
        return False
    for name in ("signal_date", "permno"):
        equal = pc.equal(left[name], right[name])
        if not bool(pc.all(pc.fill_null(equal, False)).as_py()):
            return False
    return True


def _manifest_output(
    manifest: Mapping[str, Any],
    *,
    section: str,
    name: str,
    database_root: Path,
) -> tuple[Path, Mapping[str, Any]]:
    outputs = manifest.get("outputs")
    require(isinstance(outputs, Mapping), "Stage 9 manifest outputs are missing")
    bundle = outputs.get(section)
    require(isinstance(bundle, Mapping), f"Stage 9 manifest section is missing: {section}")
    files = bundle.get("files")
    require(isinstance(files, Mapping), f"Stage 9 manifest files are missing: {section}")
    receipt = files.get(name)
    require(isinstance(receipt, Mapping), f"Stage 9 manifest omitted input: {name}")
    base = Path(str(bundle.get("path", "")))
    relative_file = Path(str(receipt.get("path", "")))
    require(
        base.parts
        and not base.is_absolute()
        and ".." not in base.parts
        and relative_file.parts
        and not relative_file.is_absolute()
        and ".." not in relative_file.parts,
        f"Unsafe Stage 9 manifest path for {name}",
    )
    path = _anchored_regular_path(
        database_root / base / relative_file,
        trusted_root=database_root,
        label=f"Stage 9 manifest input {name}",
    )
    return path, receipt


def _authenticate_pointer(
    policy: Mapping[str, Any],
    section: str,
    *,
    authority: bool,
) -> dict[str, Any]:
    specification = policy[section]
    pointer_path = _authenticate_project_file(
        specification["verified_pointer" if not authority else "verified_stage08_pointer"],
        specification[
            "verified_pointer_sha256" if not authority else "verified_stage08_pointer_sha256"
        ],
        f"{section} pointer",
    )
    pointer = _json(pointer_path)
    require(pointer.get("status") == "VERIFIED_PASSED", f"{section} pointer did not pass")
    require(
        pointer.get("build_id")
        == specification[
            "required_build_id" if not authority else "required_stage08_build_id"
        ]
        and pointer.get("verification_id")
        == specification[
            "required_verification_id" if not authority else "required_stage08_verification_id"
        ],
        f"{section} verified identity differs",
    )
    if authority:
        authorizations = pointer.get("authorizations")
        require(isinstance(authorizations, Mapping), "Stage 8 authority map is missing")
        require(
            authorizations.get(specification["required_authorization"]) is True,
            "Verified Stage 8 pointer does not authorize Stage 9",
        )
        for forbidden in (
            "stage_10_or_later_authorized",
            "models_portfolios_pnl_or_backtests_authorized",
            "validation_or_final_test_access_authorized",
        ):
            require(authorizations.get(forbidden) is False, f"Stage 8 unexpectedly authorizes {forbidden}")
    return pointer


def resolve_stage9_inputs(policy_path: Path = DEFAULT_POLICY) -> Stage9Inputs:
    """Full-hash authenticate exactly seven development-only inputs."""

    policy, _ = validate_contract(policy_path)
    methodology = policy["methodology"]
    _authenticate_project_file(methodology["source"], methodology["source_sha256"], "methodology source")
    _authenticate_project_file(
        methodology["freeze_receipt"],
        methodology["freeze_receipt_sha256"],
        "methodology freeze receipt",
    )
    _authenticate_project_file(
        policy["database"]["paths_config"],
        policy["database"]["paths_config_sha256"],
        "paths configuration",
    )
    _authenticate_project_file(
        policy["database"]["relocation_receipt"],
        policy["database"]["relocation_receipt_sha256"],
        "database relocation receipt",
    )

    authority = _authenticate_pointer(policy, "authority", authority=True)
    feature_pointer = _authenticate_pointer(policy, "verified_feature_store", authority=False)
    label_pointer = _authenticate_pointer(policy, "verified_label_store", authority=False)

    manifests: dict[str, Mapping[str, Any]] = {}
    for section, gate, pointer in (
        ("verified_feature_store", "verified_stage06_feature_store", feature_pointer),
        ("verified_label_store", "verified_stage07_label_store", label_pointer),
        ("verified_stage08_key_audit", "verified_stage08_key_audit", authority),
    ):
        specification = policy[section]
        path = _authenticate_project_file(
            specification["primary_manifest"],
            specification["primary_manifest_sha256"],
            f"{section} primary manifest",
        )
        manifest = _json(path)
        require(
            manifest.get("build_id") == pointer["build_id"],
            f"{section} manifest build differs from its verified pointer",
        )
        manifests[gate] = manifest

    declared_root = Path(policy["database"]["root"]).expanduser()
    require(not declared_root.is_symlink(), "Symlinked Stage 9 database root is prohibited")
    database_root = declared_root.resolve(strict=True)
    require(database_root.is_dir(), "Stage 9 database root is not a directory")

    table_specs = policy["inputs"]["tables"]
    tables: dict[str, SelectorInputTable] = {}
    for specification in table_specs:
        name = str(specification["name"])
        gate = str(specification["gate"])
        require(name not in tables and gate in manifests, f"Invalid Stage 9 input declaration: {name}")
        path, receipt = _manifest_output(
            manifests[gate],
            section=str(specification["manifest_section"]),
            name=name,
            database_root=database_root,
        )
        before = path.stat()
        expected_hash = str(receipt.get("sha256"))
        expected_size = receipt.get("size_bytes")
        require(
            isinstance(expected_size, int)
            and before.st_size == expected_size,
            f"Stage 9 input size differs: {name}",
        )
        require(sha256_file(path) == expected_hash, f"Stage 9 input hash differs: {name}")
        after = path.stat()
        require(_stat_identity(before) == _stat_identity(after), f"Stage 9 input changed during hashing: {name}")
        parquet = pq.ParquetFile(path)
        schema_hash = _schema_sha256(parquet.schema_arrow)
        require(
            parquet.metadata.num_rows == int(specification["rows"])
            and parquet.metadata.num_rows == int(receipt["rows"]),
            f"Stage 9 input row count differs: {name}",
        )
        require(
            schema_hash == str(specification["schema_sha256"])
            and schema_hash == str(receipt["schema_sha256"]),
            f"Stage 9 input schema differs: {name}",
        )
        tables[name] = SelectorInputTable(
            gate=gate,
            name=name,
            _path=path,
            sha256=expected_hash,
            schema_sha256=schema_hash,
            size_bytes=int(expected_size),
            rows=parquet.metadata.num_rows,
            row_groups=parquet.metadata.num_row_groups,
            predictor_permitted=bool(specification["predictor_permitted"]),
            filter_permitted=bool(specification["filter_permitted"]),
            key_audit_only=bool(specification["key_audit_only"]),
            role=str(specification["role"]),
            stat_identity=_stat_identity(after),
        )
    require(tuple(tables) == REQUIRED_INPUT_NAMES, "Resolved Stage 9 allowlist differs")
    receipts = tuple(tables[name].receipt() for name in REQUIRED_INPUT_NAMES)
    return Stage9Inputs(
        database_root=database_root,
        authority=authority,
        tables=tables,
        input_receipts=receipts,
    )


def reauthenticate_stage9_inputs(inputs: Stage9Inputs) -> None:
    require(inputs.hashes_authenticated, "Cannot reauthenticate an unauthenticated Stage 9 catalog")
    require(tuple(inputs.tables) == REQUIRED_INPUT_NAMES, "Stage 9 catalog input set changed")
    for name in REQUIRED_INPUT_NAMES:
        table = inputs.tables[name]
        path = inputs._trusted_path(name)
        before = path.stat()
        require(_stat_identity(before) == table.stat_identity, f"Stage 9 input changed: {name}")
        require(sha256_file(path) == table.sha256, f"Stage 9 input hash changed: {name}")
        after = path.stat()
        require(
            _stat_identity(before) == _stat_identity(after) == table.stat_identity,
            f"Stage 9 input changed during reauthentication: {name}",
        )


__all__ = [
    "FEATURE_SPINE_COLUMNS",
    "FIRST_SIGNAL",
    "FIRST_LONG_SIGNAL",
    "LABEL_COLUMNS",
    "LAST_SIGNAL",
    "REQUIRED_LONG_SIGNAL_DATES",
    "SPLIT_COLUMNS",
    "SelectorInputTable",
    "Stage9Inputs",
    "reauthenticate_stage9_inputs",
    "resolve_stage9_inputs",
]
