"""Read-only summaries of authenticated Stage 9 selector results.

The reporting boundary deliberately consumes only the completed-search or
primary-publication artifacts.  It does not fit a model, rewrite an artifact,
read validation/final-period payloads, or grant downstream authority.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from portfolio_bot.selectors.contract import (
    CONTRACT_ID,
    DEFAULT_POLICY,
    validate_contract,
)
from portfolio_bot.selectors.independent_verification import authenticate_primary
from portfolio_bot.selectors.publication import (
    CONFIG_NAME,
    FAMILY_ORDER,
    FIT_RECEIPTS_NAME,
    FOLD_IDS,
    FOLD_NAME,
    SEARCH_RECEIPT_NAME,
    SHORTLIST_NAME,
    open_completed_search_bundle,
    validate_search_bundle,
)


REPORT_SCHEMA_VERSION = 1
REPORT_TYPE = "stage09_selector_results_summary"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_FIT_JOB_ID = re.compile(r"stage09-fit-[0-9a-f]{64}\Z")
_FORBIDDEN_AUTHORIZATIONS = (
    "stage_10_allocation_architectures_authorized",
    "stage_11_or_later_authorized",
    "portfolios_pnl_or_backtests_authorized",
    "validation_or_final_test_access_authorized",
)
_FAMILY_LABELS = {
    "neural_network": "Neural network",
    "xgboost": "XGBoost",
    "elastic_net": "Elastic Net",
}


class SelectorReportingError(RuntimeError):
    """Raised when selector artifacts cannot be summarized safely."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SelectorReportingError(message)


def _frame(value: pa.Table | pd.DataFrame, *, label: str) -> pd.DataFrame:
    if isinstance(value, pa.Table):
        return value.to_pandas()
    if isinstance(value, pd.DataFrame):
        return value.copy()
    raise SelectorReportingError(f"{label} is not an Arrow table or DataFrame")


def _require_columns(
    frame: pd.DataFrame,
    required: set[str],
    *,
    label: str,
) -> None:
    missing = sorted(required.difference(frame.columns))
    _require(not missing, f"{label} omitted required columns: {missing}")


def _finite_float(value: Any, *, label: str) -> float:
    try:
        observed = float(value)
    except (TypeError, ValueError) as exc:
        raise SelectorReportingError(f"{label} is not numeric") from exc
    _require(math.isfinite(observed), f"{label} is not finite")
    return observed


def _integer(value: Any, *, label: str) -> int:
    observed = _finite_float(value, label=label)
    _require(observed.is_integer(), f"{label} is not an integer")
    return int(observed)


def _bool(value: Any, *, label: str) -> bool:
    _require(
        isinstance(value, bool) or type(value).__name__ in {"bool", "bool_"},
        f"{label} is not boolean",
    )
    return bool(value)


def _timing_summary(frame: pd.DataFrame) -> dict[str, int | float]:
    values = pd.to_numeric(frame["elapsed_seconds"], errors="coerce")
    _require(
        bool(values.notna().all())
        and bool((values > 0.0).all())
        and bool(values.map(math.isfinite).all()),
        "Fit elapsed_seconds contains a nonpositive or nonfinite value",
    )
    return {
        "fit_receipts": int(len(values)),
        "sum_fit_seconds": float(values.sum()),
        "min_fit_seconds": float(values.min()),
        "median_fit_seconds": float(values.median()),
        "p95_fit_seconds": float(values.quantile(0.95, interpolation="linear")),
        "max_fit_seconds": float(values.max()),
    }


def _optional_nonnegative_number(
    receipt: Mapping[str, Any],
    candidates: tuple[tuple[str, ...], ...],
) -> tuple[int | float | None, str | None]:
    for path in candidates:
        value: Any = receipt
        for part in path:
            if not isinstance(value, Mapping) or part not in value:
                break
            value = value[part]
        else:
            if isinstance(value, bool):
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(number) and number >= 0.0:
                normalized: int | float = (
                    int(number) if number.is_integer() else number
                )
                return normalized, ".".join(path)
    return None, None


def _cache_evidence(
    fits: pd.DataFrame,
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    job_ids = fits["job_id"].astype(str)
    prediction_hashes = fits["prediction_sha256"].astype(str)
    complete = fits["job_status"].astype(str).eq("COMPLETE")
    converged = fits["fit_converged"].map(
        lambda value: _bool(value, label="fit_converged")
    )
    valid_hashes = prediction_hashes.map(
        lambda value: _SHA256.fullmatch(value) is not None
    )
    content_addressed = job_ids.map(
        lambda value: _FIT_JOB_ID.fullmatch(value) is not None
    )
    bound = (
        content_addressed
        & complete
        & converged
        & valid_hashes
    )
    cache_hits, cache_path = _optional_nonnegative_number(
        receipt,
        (
            ("execution", "fit_cache_hits"),
            ("execution", "cache_hits"),
            ("fit_cache_hits",),
            ("cache_hits",),
        ),
    )
    search_elapsed, elapsed_path = _optional_nonnegative_number(
        receipt,
        (
            ("execution", "elapsed_seconds"),
            ("search_elapsed_seconds",),
            ("elapsed_seconds",),
        ),
    )
    return {
        "content_addressed_job_receipts": int(bound.sum()),
        "unique_job_ids": int(job_ids.nunique(dropna=False)),
        "prediction_sha256_receipts": int(valid_hashes.sum()),
        "complete_fit_receipts": int(complete.sum()),
        "converged_fit_receipts": int(converged.sum()),
        "all_fit_receipts_bound_and_complete": bool(
            bound.all() and job_ids.is_unique
        ),
        "cache_hits": cache_hits,
        "cache_hit_evidence": (
            f"persisted_search_receipt:{cache_path}"
            if cache_path is not None
            else "not_persisted_in_search_or_primary_artifacts"
        ),
        "recorded_search_elapsed_seconds": search_elapsed,
        "search_elapsed_evidence": (
            f"persisted_search_receipt:{elapsed_path}"
            if elapsed_path is not None
            else "not_persisted_in_search_or_primary_artifacts"
        ),
    }


def _scope_evidence(receipt: Mapping[str, Any]) -> dict[str, bool | int]:
    scope = receipt.get("scope")
    _require(isinstance(scope, Mapping), "Search receipt omitted scope evidence")
    _require("development_only" in scope, "Search receipt omitted development_only")
    development_only = _bool(
        scope["development_only"], label="scope development_only"
    )
    _require(development_only, "Search receipt is not development-only")

    counts: dict[str, int] = {}
    for name in (
        "validation_2015_2019_rows_read",
        "final_2020_2025_rows_read",
        "portfolio_weights_created",
        "trades_or_pnl_created",
    ):
        _require(name in scope, f"Search receipt omitted scope field: {name}")
        value = scope[name]
        _require(
            isinstance(value, int) and not isinstance(value, bool),
            f"Search receipt scope field is not an integer: {name}",
        )
        _require(value == 0, f"Search receipt records prohibited activity: {name}")
        counts[name] = value
    return {"development_only": development_only, **counts}


def _authorization_evidence(
    source: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> dict[str, Any]:
    value = (
        source["authorizations"]
        if "authorizations" in source
        else receipt.get("authorizations")
    )
    _require(isinstance(value, Mapping), "Source omitted authorization evidence")
    authorizations = dict(value)
    boundary: dict[str, bool] = {}
    for name in _FORBIDDEN_AUTHORIZATIONS:
        _require(name in authorizations, f"Source omitted authorization field: {name}")
        observed = _bool(authorizations[name], label=f"authorization {name}")
        _require(not observed, f"Reporting source grants prohibited authority: {name}")
        boundary[name] = observed
    return {
        "stage_10_authorized": boundary[
            "stage_10_allocation_architectures_authorized"
        ],
        "stage_11_or_later_authorized": boundary[
            "stage_11_or_later_authorized"
        ],
        "portfolios_pnl_or_backtests_authorized": boundary[
            "portfolios_pnl_or_backtests_authorized"
        ],
        "validation_or_final_test_access_authorized": boundary[
            "validation_or_final_test_access_authorized"
        ],
        "source_authorizations": authorizations,
    }


def _shortlist_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    records: list[dict[str, Any]] = []
    for row in frame.sort_values("shortlist_order", kind="mergesort").itertuples(
        index=False
    ):
        records.append(
            {
                "shortlist_order": _integer(
                    row.shortlist_order, label="shortlist_order"
                ),
                "config_id": str(row.config_id),
                "selection_reasons": str(row.selection_reasons).split("|"),
                "benchmark_only_fallback": _bool(
                    row.benchmark_only_fallback,
                    label="benchmark_only_fallback",
                ),
                "gate_pass": _bool(row.gate_pass, label="shortlist gate_pass"),
                "mean_rank_ic": _finite_float(
                    row.mean_rank_ic, label="shortlist mean_rank_ic"
                ),
                "rank_ic_bootstrap_se": _finite_float(
                    row.rank_ic_bootstrap_se,
                    label="shortlist rank_ic_bootstrap_se",
                ),
                "positive_fold_count": _integer(
                    row.positive_fold_count,
                    label="shortlist positive_fold_count",
                ),
                "fold_dispersion": _finite_float(
                    row.fold_dispersion, label="shortlist fold_dispersion"
                ),
                "complexity": _finite_float(
                    row.complexity, label="shortlist complexity"
                ),
                "family_best_mean_rank_ic": _finite_float(
                    row.family_best_mean_rank_ic,
                    label="shortlist family_best_mean_rank_ic",
                ),
                "family_best_bootstrap_se": _finite_float(
                    row.family_best_bootstrap_se,
                    label="shortlist family_best_bootstrap_se",
                ),
                "one_se_threshold": _finite_float(
                    row.one_se_threshold, label="shortlist one_se_threshold"
                ),
            }
        )
    return records


def _best_row(
    family: pd.DataFrame,
    shortlist: pd.DataFrame,
) -> pd.Series:
    maximum = _finite_float(
        pd.to_numeric(family["mean_rank_ic"], errors="coerce").max(),
        label="family maximum mean_rank_ic",
    )
    tied = family.loc[
        pd.to_numeric(family["mean_rank_ic"], errors="coerce")
        >= maximum - 1e-12
    ]
    if not shortlist.empty:
        official = shortlist.loc[
            shortlist["selection_reasons"]
            .astype(str)
            .map(
                lambda value: "highest_mean_rank_ic" in value.split("|")
                or "benchmark_only_fallback" in value.split("|")
            )
        ]
        official = official.loc[
            official["config_id"].astype(str).isin(tied["config_id"].astype(str))
        ]
        if not official.empty:
            config_id = str(
                official.sort_values("shortlist_order", kind="mergesort")
                .iloc[0]["config_id"]
            )
            return tied.loc[tied["config_id"].astype(str).eq(config_id)].iloc[0]
    return tied.sort_values("config_id", kind="mergesort").iloc[0]


def _best_record(
    row: pd.Series,
    folds: pd.DataFrame,
) -> dict[str, Any]:
    config_id = str(row["config_id"])
    selected_folds = folds.loc[folds["config_id"].astype(str).eq(config_id)].copy()
    _require(
        set(selected_folds["fold_id"].astype(str)) == set(FOLD_IDS)
        and len(selected_folds) == len(FOLD_IDS),
        f"Best configuration {config_id} does not have four exact fold metrics",
    )
    fold_order = {fold_id: index for index, fold_id in enumerate(FOLD_IDS)}
    selected_folds["_order"] = selected_folds["fold_id"].map(fold_order)
    fold_records = [
        {
            "fold_id": str(fold.fold_id),
            "months": _integer(fold.months, label="fold months"),
            "mean_rank_ic": _finite_float(
                fold.mean_rank_ic, label="fold mean_rank_ic"
            ),
            "mean_linear_ic": _finite_float(
                fold.mean_linear_ic, label="fold mean_linear_ic"
            ),
        }
        for fold in selected_folds.sort_values("_order", kind="mergesort").itertuples(
            index=False
        )
    ]
    return {
        "config_id": config_id,
        "mean_rank_ic": _finite_float(
            row["mean_rank_ic"], label="best mean_rank_ic"
        ),
        "mean_linear_ic": _finite_float(
            row["mean_linear_ic"], label="best mean_linear_ic"
        ),
        "rank_ic_bootstrap_se": _finite_float(
            row["rank_ic_bootstrap_se"], label="best rank_ic_bootstrap_se"
        ),
        "linear_ic_bootstrap_se": _finite_float(
            row["linear_ic_bootstrap_se"],
            label="best linear_ic_bootstrap_se",
        ),
        "positive_fold_count": _integer(
            row["positive_fold_count"], label="best positive_fold_count"
        ),
        "fold_dispersion": _finite_float(
            row["fold_dispersion"], label="best fold_dispersion"
        ),
        "gate_pass": _bool(row["gate_pass"], label="best gate_pass"),
        "complexity": _finite_float(row["complexity"], label="best complexity"),
        "trial_phase": str(row["trial_phase"]),
        "fold_metrics": fold_records,
    }


def build_selector_results_summary(
    tables: Mapping[str, pa.Table | pd.DataFrame],
    *,
    receipt: Mapping[str, Any],
    source: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a JSON-safe summary from already authenticated compact tables.

    The function is intentionally pure and performs no I/O.  Artifact readers
    below authenticate their source before calling it; tests and notebooks can
    also use it with in-memory tables.
    """

    required_tables = {CONFIG_NAME, FOLD_NAME, SHORTLIST_NAME, FIT_RECEIPTS_NAME}
    _require(
        required_tables.issubset(tables),
        f"Selector summary table set is incomplete: {sorted(required_tables.difference(tables))}",
    )
    configurations = _frame(tables[CONFIG_NAME], label=CONFIG_NAME)
    folds = _frame(tables[FOLD_NAME], label=FOLD_NAME)
    shortlist = _frame(tables[SHORTLIST_NAME], label=SHORTLIST_NAME)
    fits = _frame(tables[FIT_RECEIPTS_NAME], label=FIT_RECEIPTS_NAME)
    _require_columns(
        configurations,
        {
            "model_family",
            "config_id",
            "mean_rank_ic",
            "mean_linear_ic",
            "rank_ic_bootstrap_se",
            "linear_ic_bootstrap_se",
            "positive_fold_count",
            "fold_dispersion",
            "gate_pass",
            "complexity",
            "trial_phase",
        },
        label=CONFIG_NAME,
    )
    _require_columns(
        folds,
        {
            "model_family",
            "config_id",
            "fold_id",
            "months",
            "mean_rank_ic",
            "mean_linear_ic",
        },
        label=FOLD_NAME,
    )
    _require_columns(
        shortlist,
        {
            "model_family",
            "config_id",
            "shortlist_order",
            "selection_reasons",
            "benchmark_only_fallback",
            "gate_pass",
            "mean_rank_ic",
            "rank_ic_bootstrap_se",
            "positive_fold_count",
            "fold_dispersion",
            "complexity",
            "family_best_mean_rank_ic",
            "family_best_bootstrap_se",
            "one_se_threshold",
        },
        label=SHORTLIST_NAME,
    )
    _require_columns(
        fits,
        {
            "model_family",
            "job_id",
            "job_status",
            "prediction_sha256",
            "fit_converged",
            "elapsed_seconds",
        },
        label=FIT_RECEIPTS_NAME,
    )
    families = set(configurations["model_family"].astype(str))
    _require(
        families == set(FAMILY_ORDER),
        "Selector summary requires the three frozen model families",
    )
    _require(
        not configurations.duplicated(["model_family", "config_id"]).any(),
        "Configuration summary key is duplicated",
    )

    family_records: list[dict[str, Any]] = []
    for model_family in FAMILY_ORDER:
        family_config = configurations.loc[
            configurations["model_family"].astype(str).eq(model_family)
        ].copy()
        family_folds = folds.loc[
            folds["model_family"].astype(str).eq(model_family)
        ].copy()
        family_shortlist = shortlist.loc[
            shortlist["model_family"].astype(str).eq(model_family)
        ].copy()
        family_fits = fits.loc[
            fits["model_family"].astype(str).eq(model_family)
        ].copy()
        _require(not family_fits.empty, f"{model_family} has no fit receipts")
        gates = family_config["gate_pass"].map(
            lambda value: _bool(value, label=f"{model_family} gate_pass")
        )
        best = _best_row(family_config, family_shortlist)
        family_records.append(
            {
                "model_family": model_family,
                "configuration_count": int(len(family_config)),
                "gate_counts": {
                    "passed": int(gates.sum()),
                    "failed": int((~gates).sum()),
                },
                "best_configuration": _best_record(best, family_folds),
                "shortlist": _shortlist_records(family_shortlist),
                "fit_timing": _timing_summary(family_fits),
            }
        )

    all_gates = configurations["gate_pass"].map(
        lambda value: _bool(value, label="configuration gate_pass")
    )
    scope = _scope_evidence(receipt)
    authorization_boundary = _authorization_evidence(source, receipt)
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "report_type": REPORT_TYPE,
        "contract_id": CONTRACT_ID,
        "status": "READ_ONLY_SUMMARY",
        "source": dict(source),
        "scope": scope,
        "gate_counts": {
            "configurations": int(len(configurations)),
            "passed": int(all_gates.sum()),
            "failed": int((~all_gates).sum()),
        },
        "shortlist_count": int(len(shortlist)),
        "families": family_records,
        "fit_timing": _timing_summary(fits),
        "cache_evidence": _cache_evidence(fits, receipt),
        "authorization_boundary": authorization_boundary,
    }


def summarize_completed_search_bundle(
    search_bundle_path: Path,
    *,
    policy_path: Path = DEFAULT_POLICY,
) -> dict[str, Any]:
    """Authenticate and summarize one sealed completed-search directory."""

    policy, _ = validate_contract(policy_path)
    database_root = Path(str(policy["database"]["root"])).expanduser().resolve()
    with open_completed_search_bundle(
        search_bundle_path,
        database_root=database_root,
    ) as (identity, guard):
        tables, _checks = validate_search_bundle(identity, guard)
        source = {
            "kind": "completed_search_bundle",
            "path": str(identity.path),
            "status": str(identity.receipt["status"]),
            "search_id": identity.search_id,
            "search_receipt_sha256": identity.receipt_sha256,
            "authorizations": dict(identity.receipt.get("authorizations", {})),
        }
        summary = build_selector_results_summary(
            tables,
            receipt=identity.receipt,
            source=source,
        )
        guard.gate()
        return summary


def _read_guarded_table(guard: Any, name: str) -> pa.Table:
    with guard.open_binary(name) as stream:
        return pq.read_table(stream, use_threads=False).combine_chunks()


def _read_guarded_json(guard: Any, name: str) -> dict[str, Any]:
    with guard.open_binary(name) as stream:
        payload = json.loads(stream.read())
    _require(isinstance(payload, dict), f"{name} is not a JSON object")
    return payload


def summarize_primary_pointer(
    primary_pointer_path: Path | None = None,
    *,
    policy_path: Path = DEFAULT_POLICY,
) -> dict[str, Any]:
    """Authenticate and summarize one primary pointer and its closed outputs."""

    with authenticate_primary(
        policy_path=policy_path,
        primary_pointer_path=primary_pointer_path,
    ) as artifacts:
        pointer = json.loads(artifacts.pointer_payload)
        _require(isinstance(pointer, dict), "Primary pointer is not a JSON object")
        interim = artifacts.output_guards["interim"]
        processed = artifacts.output_guards["processed"]
        tables = {
            FIT_RECEIPTS_NAME: _read_guarded_table(interim, FIT_RECEIPTS_NAME),
            CONFIG_NAME: _read_guarded_table(processed, CONFIG_NAME),
            FOLD_NAME: _read_guarded_table(processed, FOLD_NAME),
            SHORTLIST_NAME: _read_guarded_table(processed, SHORTLIST_NAME),
        }
        receipt = _read_guarded_json(processed, SEARCH_RECEIPT_NAME)
        source = {
            "kind": "primary_pointer",
            "path": str(artifacts.pointer_path),
            "status": str(pointer["status"]),
            "build_id": str(pointer["build_id"]),
            "pointer_sha256": artifacts.pointer_sha256,
            "primary_manifest_sha256": artifacts.manifest_sha256,
            "search_id": str(artifacts.manifest["inputs"]["search_id"]),
            "search_receipt_sha256": str(
                artifacts.manifest["bindings"]["search_receipt_sha256"]
            ),
            "authorizations": dict(pointer["authorizations"]),
        }
        summary = build_selector_results_summary(
            tables,
            receipt=receipt,
            source=source,
        )
        artifacts.gate()
        return summary


def summarize_selector_results(
    *,
    search_bundle_path: Path | None = None,
    primary_pointer_path: Path | None = None,
    policy_path: Path = DEFAULT_POLICY,
) -> dict[str, Any]:
    """Dispatch to exactly one supported authenticated Stage 9 source."""

    _require(
        (search_bundle_path is None) != (primary_pointer_path is None),
        "Provide exactly one of search_bundle_path or primary_pointer_path",
    )
    if search_bundle_path is not None:
        return summarize_completed_search_bundle(
            search_bundle_path,
            policy_path=policy_path,
        )
    return summarize_primary_pointer(
        primary_pointer_path,
        policy_path=policy_path,
    )


def _md(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _metric(value: Any) -> str:
    return f"{float(value):.6f}"


def render_selector_results_markdown(summary: Mapping[str, Any]) -> str:
    """Render a stable human-readable view of a selector summary."""

    _require(
        summary.get("report_type") == REPORT_TYPE,
        "Markdown input is not a Stage 9 selector summary",
    )
    _require(
        summary.get("status") == "READ_ONLY_SUMMARY",
        "Markdown input is not a read-only selector summary",
    )
    source = summary["source"]
    gate = summary["gate_counts"]
    boundary = summary["authorization_boundary"]
    for name in (
        "stage_10_authorized",
        "stage_11_or_later_authorized",
        "portfolios_pnl_or_backtests_authorized",
        "validation_or_final_test_access_authorized",
    ):
        _require(name in boundary, f"Markdown input omitted authorization field: {name}")
        _require(
            not _bool(boundary[name], label=f"Markdown authorization {name}"),
            f"Markdown input grants prohibited authority: {name}",
        )
    lines = [
        "# Stage 9 selector results summary",
        "",
        f"- Source: `{_md(source['kind'])}` — `{_md(source['path'])}`",
        f"- Source status: `{_md(source['status'])}`",
        f"- Configurations: {gate['configurations']} "
        f"({gate['passed']} gate pass, {gate['failed']} gate fail)",
        f"- Shortlisted configurations: {summary['shortlist_count']}",
        "",
        "## Family overview",
        "",
        "| Family | Configurations | Gate pass | Gate fail | Best config | Mean rank IC | Mean linear IC | Shortlist |",
        "|---|---:|---:|---:|---|---:|---:|---:|",
    ]
    for family in summary["families"]:
        best = family["best_configuration"]
        counts = family["gate_counts"]
        lines.append(
            "| "
            f"{_md(_FAMILY_LABELS.get(family['model_family'], family['model_family']))} | "
            f"{family['configuration_count']} | {counts['passed']} | {counts['failed']} | "
            f"`{_md(best['config_id'])}` | {_metric(best['mean_rank_ic'])} | "
            f"{_metric(best['mean_linear_ic'])} | {len(family['shortlist'])} |"
        )

    lines.extend(["", "## Best-configuration fold metrics", ""])
    for family in summary["families"]:
        best = family["best_configuration"]
        label = _FAMILY_LABELS.get(family["model_family"], family["model_family"])
        lines.extend(
            [
                f"### {_md(label)} — `{_md(best['config_id'])}`",
                "",
                "| Fold | Months | Mean rank IC | Mean linear IC |",
                "|---|---:|---:|---:|",
            ]
        )
        for fold in best["fold_metrics"]:
            lines.append(
                f"| `{_md(fold['fold_id'])}` | {fold['months']} | "
                f"{_metric(fold['mean_rank_ic'])} | {_metric(fold['mean_linear_ic'])} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Shortlist",
            "",
            "| Family | Order | Config | Reasons | Gate | Mean rank IC | One-SE threshold | Benchmark fallback |",
            "|---|---:|---|---|---|---:|---:|---|",
        ]
    )
    shortlist_rows = 0
    for family in summary["families"]:
        label = _FAMILY_LABELS.get(family["model_family"], family["model_family"])
        for row in family["shortlist"]:
            shortlist_rows += 1
            reasons = ", ".join(row["selection_reasons"])
            lines.append(
                f"| {_md(label)} | {row['shortlist_order']} | `{_md(row['config_id'])}` | "
                f"{_md(reasons)} | {'pass' if row['gate_pass'] else 'fail'} | "
                f"{_metric(row['mean_rank_ic'])} | {_metric(row['one_se_threshold'])} | "
                f"{'yes' if row['benchmark_only_fallback'] else 'no'} |"
            )
    if shortlist_rows == 0:
        lines.append("| — | — | — | No configurations shortlisted | — | — | — | — |")

    lines.extend(
        [
            "",
            "## Fit timing and cache evidence",
            "",
            "| Family | Fit receipts | Sum fit seconds | Median | P95 | Maximum |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for family in summary["families"]:
        timing = family["fit_timing"]
        label = _FAMILY_LABELS.get(family["model_family"], family["model_family"])
        lines.append(
            f"| {_md(label)} | {timing['fit_receipts']} | "
            f"{timing['sum_fit_seconds']:.3f} | {timing['median_fit_seconds']:.3f} | "
            f"{timing['p95_fit_seconds']:.3f} | {timing['max_fit_seconds']:.3f} |"
        )
    evidence = summary["cache_evidence"]
    cache_hits = (
        "not recorded"
        if evidence["cache_hits"] is None
        else str(evidence["cache_hits"])
    )
    lines.extend(
        [
            "",
            f"- Content-addressed, complete fit receipts: "
            f"{evidence['content_addressed_job_receipts']}/"
            f"{summary['fit_timing']['fit_receipts']}",
            f"- Unique job IDs: {evidence['unique_job_ids']}",
            f"- Recorded cache hits: {cache_hits} "
            f"(`{_md(evidence['cache_hit_evidence'])}`)",
            "",
            "## Authorization boundary",
            "",
            f"- Stage 10 authorized by this source: "
            f"{'yes' if boundary['stage_10_authorized'] else 'no'}",
            f"- Stage 11 or later authorized: "
            f"{'yes' if boundary['stage_11_or_later_authorized'] else 'no'}",
            f"- Portfolios, P&L, or backtests authorized: "
            f"{'yes' if boundary['portfolios_pnl_or_backtests_authorized'] else 'no'}",
            "",
        ]
    )
    return "\n".join(lines)


__all__ = [
    "REPORT_SCHEMA_VERSION",
    "REPORT_TYPE",
    "SelectorReportingError",
    "build_selector_results_summary",
    "render_selector_results_markdown",
    "summarize_completed_search_bundle",
    "summarize_primary_pointer",
    "summarize_selector_results",
]
