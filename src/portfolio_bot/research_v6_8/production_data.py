"""Production-data source audit for the frozen V6.8 paper strategy.

The module is intentionally read-only.  It maps every registered feature to
one explicit live-source policy and emits only secret-free readiness metadata.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from portfolio_bot.research_v6_8.credentials import paper_credential_presence


class ProductionDataContractError(RuntimeError):
    """The feature-source contract does not match the frozen registry."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _expand_ranges(ranges: list[list[int]]) -> list[int]:
    values: list[int] = []
    for start, end in ranges:
        if start > end:
            raise ProductionDataContractError(f"invalid ordinal range {start}-{end}")
        values.extend(range(int(start), int(end) + 1))
    return values


def build_feature_source_matrix(
    contract: Mapping[str, Any], registry: pd.DataFrame
) -> pd.DataFrame:
    """Bind every frozen feature ordinal to exactly one production source group."""

    required = {
        "model_ordinal",
        "feature_name",
        "family",
        "source",
        "source_column",
        "transformation",
    }
    missing_columns = sorted(required - set(registry.columns))
    if missing_columns:
        raise ProductionDataContractError(
            f"registry is missing required columns: {', '.join(missing_columns)}"
        )
    rows: list[dict[str, Any]] = []
    assignments: dict[int, str] = {}
    for group in contract["feature_groups"]:
        ordinals = _expand_ranges(group["ordinal_ranges"])
        if len(ordinals) != int(group["feature_count"]):
            raise ProductionDataContractError(
                f"{group['id']} declares {group['feature_count']} features but ranges contain {len(ordinals)}"
            )
        for ordinal in ordinals:
            if ordinal in assignments:
                raise ProductionDataContractError(
                    f"feature ordinal {ordinal} assigned to both {assignments[ordinal]} and {group['id']}"
                )
            assignments[ordinal] = str(group["id"])
            match = registry.loc[registry["model_ordinal"] == ordinal]
            if len(match) != 1:
                raise ProductionDataContractError(
                    f"feature ordinal {ordinal} appears {len(match)} times in registry"
                )
            feature = match.iloc[0]
            rows.append(
                {
                    "model_ordinal": ordinal,
                    "feature_name": feature["feature_name"],
                    "family": feature["family"],
                    "historical_registry_source": feature["source"],
                    "historical_source_column": feature["source_column"],
                    "transformation": feature["transformation"],
                    "production_group": group["id"],
                    "production_group_label": group["label"],
                    "refresh_class": group["refresh_class"],
                    "primary_source": group["primary_source"],
                    "current_state": group["current_state"],
                    "fallback": group["fallback"],
                    "required_action": group["action"],
                    "yfinance_primary_allowed": bool(
                        contract["production_policy"]["yfinance_primary_allowed"]
                    ),
                }
            )
    expected = set(int(value) for value in registry["model_ordinal"])
    observed = set(assignments)
    if expected != observed:
        raise ProductionDataContractError(
            f"feature assignment mismatch; missing={sorted(expected-observed)}, extra={sorted(observed-expected)}"
        )
    result = pd.DataFrame(rows).sort_values("model_ordinal").reset_index(drop=True)
    if len(result) != int(contract["frozen_feature_registry"]["feature_count"]):
        raise ProductionDataContractError("matrix row count differs from frozen contract")
    return result


def summarize_feature_sources(matrix: pd.DataFrame) -> pd.DataFrame:
    """Produce one decision-oriented row per source group."""

    columns = [
        "production_group",
        "production_group_label",
        "refresh_class",
        "primary_source",
        "current_state",
        "required_action",
    ]
    return (
        matrix.groupby(columns, dropna=False, sort=False)
        .agg(
            feature_count=("feature_name", "size"),
            first_ordinal=("model_ordinal", "min"),
            last_ordinal=("model_ordinal", "max"),
        )
        .reset_index()
    )


def _month_files(path: Path, pattern: str) -> list[str]:
    months: list[str] = []
    for item in path.glob(pattern):
        match = re.search(r"(\d{4}-\d{2})", item.name)
        if match:
            months.append(match.group(1))
    return sorted(months)


def _process_running(pattern: str) -> bool:
    try:
        result = subprocess.run(
            ["pgrep", "-f", pattern],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0 and bool(result.stdout.strip())


def local_source_readiness(
    contract: Mapping[str, Any], *, checked_at: datetime | None = None
) -> dict[str, Any]:
    """Inspect local credentials, archives, and licensed-session markers.

    No network calls are made and credential values are never returned.
    """

    checked = checked_at or datetime.now(timezone.utc)
    env = contract["environment"]
    legacy = contract["legacy_local_sources"]
    credential_presence = paper_credential_presence(
        key_environment=env["alpaca_key_environment"],
        secret_environment=env["alpaca_secret_environment"],
    )
    alpaca_key = bool(credential_presence["key"])
    alpaca_secret = bool(credential_presence["secret"])
    news_months = _month_files(Path(legacy["alpaca_news_directory"]), "alpaca_news_????-??.manifest.json")
    lseg_months = _month_files(Path(legacy["lseg_directory"]) / "receipts", "????-??.json")
    lseg_usd_months = _month_files(Path(legacy["lseg_usd_directory"]) / "receipts", "????-??.json")
    home = Path.home()
    wrds_present = any(
        [
            (home / ".pgpass").exists(),
            (home / ".wrds").exists(),
            bool(os.environ.get("WRDS_USERNAME")),
        ]
    )
    files = {
        name: {
            "exists": Path(path).exists(),
            "modified_at": (
                datetime.fromtimestamp(Path(path).stat().st_mtime, tz=timezone.utc).isoformat()
                if Path(path).exists()
                else None
            ),
        }
        for name, path in legacy.items()
        if name not in {"alpaca_news_directory", "lseg_directory", "lseg_usd_directory"}
    }
    capability_receipt_path = Path(env["lseg_capability_receipt"])
    capability_receipt: dict[str, Any] = {}
    if capability_receipt_path.is_file():
        try:
            capability_receipt = json.loads(
                capability_receipt_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            capability_receipt = {}
    capability_status = capability_receipt.get("status")
    capability_pass = capability_status == (
        "CAPABILITY_AND_SAME_SOURCE_REPLAY_PASS_PROVIDER_BRIDGE_PENDING"
    )
    market_parity_path = Path(env.get("lseg_market_parity_receipt", ""))
    market_parity_receipt: dict[str, Any] = {}
    if market_parity_path.is_file():
        try:
            market_parity_receipt = json.loads(
                market_parity_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            market_parity_receipt = {}
    market_parity_pass = bool(
        market_parity_receipt.get("status") == "PASS"
        and market_parity_receipt.get("months_completed")
        == market_parity_receipt.get("months_expected")
        and float(
            market_parity_receipt.get(
                "all_month_resolved_instrument_share", 0.0
            )
        )
        == 1.0
    )
    derived_parity_path = Path(
        env.get("lseg_market_derived_parity_receipt", "")
    )
    derived_parity_receipt: dict[str, Any] = {}
    if derived_parity_path.is_file():
        try:
            derived_parity_receipt = json.loads(
                derived_parity_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            derived_parity_receipt = {}
    derived_features_tested = int(derived_parity_receipt.get("features_tested", 0))
    derived_features_passing = int(
        derived_parity_receipt.get("features_passing_feature_level_gate", 0)
    )
    derived_features_failing = int(
        derived_parity_receipt.get("features_failing_feature_level_gate", 0)
    )
    derived_complete = bool(
        derived_parity_receipt.get("months") == 20
        and derived_features_tested == 47
        and derived_features_passing + derived_features_failing == 47
    )
    # The live policy is now explicit: parity-passing LSEG market formulas are
    # used, non-equivalent provider formulas and the Compustat/13F/OptionMetrics
    # blocks follow the frozen missing-indicator path, and official lagged
    # factor data are used without a same-month nowcast.  Those are disclosed
    # information states, not unfinished source substitutions.  A trade remains
    # blocked until one current package authenticates every group.
    blockers = ["a current identity-bound 318-feature/850-column paper signal package has not passed its producer gates"]
    if not alpaca_key or not alpaca_secret:
        blockers.append("rotated Alpaca credentials are not present in the configured environment")
    if not capability_pass:
        blockers.append("the LSEG production capability and same-source replay probe has not passed")
    if not market_parity_pass:
        blockers.append(
            "the 2024-01 through 2025-08 sampled CRSP-to-LSEG market primitive and identifier gate has not passed"
        )
    return {
        "schema": "research_v6_8_production_data_readiness_v4",
        "checked_at": checked.astimezone(timezone.utc).isoformat(),
        "status": "BLOCKED_FOR_PAPER_ORDERS",
        "feature_schema_complete": True,
        "frozen_feature_count": int(contract["frozen_feature_registry"]["feature_count"]),
        "provider_checks": {
            "alpaca_credentials_present": alpaca_key and alpaca_secret,
            "lseg_workspace_process_running": _process_running("Refinitiv Workspace"),
            "lseg_capability_probe_pass": capability_pass,
            "lseg_capability_probe_status": capability_status,
            "lseg_market_parity_pass": market_parity_pass,
            "lseg_market_parity_status": market_parity_receipt.get("status"),
            "lseg_market_parity_months": market_parity_receipt.get(
                "months_completed"
            ),
            "lseg_market_parity_resolved_share": market_parity_receipt.get(
                "all_month_resolved_instrument_share"
            ),
            "lseg_market_derived_audit_complete": derived_complete,
            "lseg_market_derived_features_tested": derived_features_tested,
            "lseg_market_derived_features_passing": derived_features_passing,
            "lseg_market_derived_features_failing": derived_features_failing,
            "wrds_credentials_present": wrds_present,
            "legacy_alpaca_news_manifest_months": len(news_months),
            "legacy_alpaca_news_latest_month": news_months[-1] if news_months else None,
            "legacy_lseg_receipt_months": len(lseg_months),
            "legacy_lseg_latest_month": lseg_months[-1] if lseg_months else None,
            "legacy_lseg_usd_receipt_months": len(lseg_usd_months),
            "legacy_lseg_usd_latest_month": lseg_usd_months[-1] if lseg_usd_months else None,
        },
        "legacy_files": files,
        "blockers": blockers,
        "secrets_in_receipt": False,
    }


def audit_production_sources(
    config_path: Path, output_directory: Path
) -> dict[str, Any]:
    """Validate the contract and atomically publish a secret-free audit bundle."""

    contract = json.loads(config_path.read_text("utf-8"))
    registry_binding = contract["frozen_feature_registry"]
    registry_path = Path(registry_binding["path"])
    if _sha256(registry_path) != registry_binding["sha256"]:
        raise ProductionDataContractError("frozen feature-registry identity mismatch")
    registry = pd.read_parquet(registry_path)
    matrix = build_feature_source_matrix(contract, registry)
    summary = summarize_feature_sources(matrix)
    readiness = local_source_readiness(contract)
    class_counts = {
        str(key): int(value)
        for key, value in matrix.groupby("refresh_class").size().to_dict().items()
    }
    readiness["refresh_class_feature_counts"] = class_counts
    readiness["source_group_count"] = int(summary.shape[0])
    readiness["feature_matrix_rows"] = int(matrix.shape[0])
    readiness["yfinance_primary_allowed"] = False

    output_directory.mkdir(parents=True, exist_ok=True)
    payloads = {
        "feature_source_matrix.csv": matrix.to_csv(index=False).encode("utf-8"),
        "source_group_summary.csv": summary.to_csv(index=False).encode("utf-8"),
        "readiness.json": (json.dumps(readiness, indent=2, sort_keys=True) + "\n").encode("utf-8"),
    }
    for name, data in payloads.items():
        target = output_directory / name
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_bytes(data)
        temporary.replace(target)
    return readiness


__all__ = [
    "ProductionDataContractError",
    "audit_production_sources",
    "build_feature_source_matrix",
    "local_source_readiness",
    "summarize_feature_sources",
]
