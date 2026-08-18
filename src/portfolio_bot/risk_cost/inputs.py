"""Authenticate and deliver the closed, development-only Stage 8 inputs."""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from portfolio_bot.features.label_store_inputs import (
    LabelStoreInputs,
    resolve_label_input_catalog,
)
from portfolio_bot.risk_cost.contract import (
    DEFAULT_POLICY,
    PROJECT_ROOT,
    RiskCostContractError,
    load_policy,
    project_path,
    require,
    sha256_file,
    validate_contract,
)


LAST_SIGNAL = dt.date(2014, 10, 31)

FEATURE_COLUMNS = (
    "signal_date",
    "permno",
    "permco",
    "long_eligible",
    "short_eligible",
    "primary_exchange",
    "ff12_sector_category",
    "domicile_country_category",
    "median_dollar_volume_20d",
    "median_estimated_full_spread_20d",
)


@dataclass(frozen=True)
class Stage8Inputs:
    """Authenticated input catalog with bounded read methods only."""

    database_root: Path
    upstream: LabelStoreInputs
    authority: Mapping[str, Any]
    input_receipts: tuple[Mapping[str, Any], ...]

    def read_feature_spine(self) -> pa.Table:
        path = self.upstream.path("feature_store_raw.parquet")
        table = pq.read_table(path, columns=list(FEATURE_COLUMNS), use_threads=False)
        require(table.num_rows == 1_263_048, "Stage 8 feature spine row count differs")
        require(
            pc.min(table["signal_date"]).as_py() == dt.date(2000, 1, 31)
            and pc.max(table["signal_date"]).as_py() == LAST_SIGNAL,
            "Stage 8 feature spine escaped development scope",
        )
        return table

    def read_rebalance_calendar(self) -> pa.Table:
        table = self.upstream.read_development_calendar()
        require(table.num_rows == 178, "Stage 8 rebalance-calendar scope differs")
        return table

    def read_trading_calendar(self) -> pa.Table:
        table = self.upstream.read_development_trading_calendar()
        mask = pc.less_equal(table["calendar_date"], pa.scalar(LAST_SIGNAL))
        bounded = table.filter(pc.fill_null(mask, False))
        require(
            bounded.num_rows > 0
            and pc.max(bounded["calendar_date"]).as_py() == LAST_SIGNAL,
            "Stage 8 trading calendar does not end at the last signal cutoff",
        )
        return bounded

    def read_factors(self) -> pa.Table:
        table = self.upstream.read_development_factors()
        dates = table["observation_date"]
        mask = pc.and_(
            pc.less_equal(dates, pa.scalar(LAST_SIGNAL)),
            pc.less_equal(table["public_availability_date"], pa.scalar(LAST_SIGNAL)),
        )
        bounded = table.filter(pc.fill_null(mask, False))
        require(
            bounded.num_rows > 0
            and pc.max(bounded["observation_date"]).as_py() == LAST_SIGNAL,
            "Stage 8 factor data do not end at the last signal cutoff",
        )
        return bounded

    def iter_return_batches(self, *, batch_rows: int = 262_144) -> Iterator[pa.RecordBatch]:
        """Yield the authenticated Stage 7 return projection.

        The upstream delivery is bounded through the label-specific 2014-12-01
        date.  Stage 8 materialization must predicate-filter every batch again
        to its own 2014-10-31 maximum and then to each signal date.
        """

        yield from self.upstream.iter_development_return_batches(batch_rows=batch_rows)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text("utf-8"))
    require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value


def _authenticate_bound_file(path_value: object, expected: object, label: str) -> Path:
    path = project_path(str(path_value))
    require(path.is_file() and not path.is_symlink(), f"Unsafe Stage 8 {label} path")
    require(sha256_file(path) == str(expected), f"Stage 8 {label} hash differs")
    return path


def _receipt(table: Any) -> dict[str, Any]:
    return {
        "gate": str(table.gate),
        "name": str(table.name),
        "sha256": str(table.sha256),
        "schema_sha256": str(table.schema_sha256),
        "size_bytes": int(table.size_bytes),
        "rows": int(table.rows),
        "row_groups": int(table.row_groups),
    }


def resolve_stage8_inputs(policy_path: Path = DEFAULT_POLICY) -> Stage8Inputs:
    policy, _ = validate_contract(policy_path)
    methodology = policy["methodology"]
    _authenticate_bound_file(
        methodology["source"], methodology["source_sha256"], "methodology source"
    )
    _authenticate_bound_file(
        methodology["freeze_receipt"],
        methodology["freeze_receipt_sha256"],
        "methodology freeze receipt",
    )

    authority_policy = policy["authority"]
    pointer_path = _authenticate_bound_file(
        authority_policy["verified_stage07_pointer"],
        authority_policy["verified_stage07_pointer_sha256"],
        "Stage 7 verified pointer",
    )
    authority = _json(pointer_path)
    require(
        authority.get("status") == "VERIFIED_PASSED"
        and authority.get("build_id") == authority_policy["required_stage07_build_id"]
        and authority.get("verification_id")
        == authority_policy["required_stage07_verification_id"],
        "Stage 7 verified authority identity differs",
    )
    authorizations = authority.get("authorizations", {})
    require(
        authorizations.get(authority_policy["required_authorization"]) is True,
        "Stage 7 does not authorize Stage 8",
    )
    for forbidden in (
        "stage_09_model_training_authorized",
        "models_portfolios_pnl_or_backtests_authorized",
        "validation_or_final_test_access_authorized",
    ):
        require(authorizations.get(forbidden) is False, f"Unexpected upstream {forbidden}")

    feature = policy["verified_feature_store"]
    feature_pointer_path = _authenticate_bound_file(
        feature["verified_pointer"], feature["verified_pointer_sha256"], "feature pointer"
    )
    feature_pointer = _json(feature_pointer_path)
    require(
        feature_pointer.get("status") == "VERIFIED_PASSED"
        and feature_pointer.get("build_id") == feature["required_build_id"]
        and feature_pointer.get("verification_id") == feature["required_verification_id"],
        "Verified feature-store identity differs",
    )
    _authenticate_bound_file(
        feature["primary_manifest"],
        feature["primary_manifest_sha256"],
        "feature manifest",
    )

    inputs_policy = policy["inputs"]
    stage7_policy = _authenticate_bound_file(
        inputs_policy["stage07_input_policy"],
        inputs_policy["stage07_input_policy_sha256"],
        "upstream input policy",
    )
    upstream = resolve_label_input_catalog(stage7_policy)
    required = tuple(str(value) for value in inputs_policy["required_table_names"])
    require(set(upstream.tables) == set(required), "Stage 8 closed input allowlist differs")
    require(
        upstream.database_root == Path(policy["database"]["root"]).resolve(),
        "Stage 8 database root differs",
    )
    receipts = tuple(_receipt(upstream.tables[name]) for name in required)
    return Stage8Inputs(
        database_root=upstream.database_root,
        upstream=upstream,
        authority=authority,
        input_receipts=receipts,
    )
