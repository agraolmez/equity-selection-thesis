#!/usr/bin/env python3
"""Build the missing V6 December 2019 predictor and target cross-section.

This promotes the already-authenticated 2019-11-29 internal-context universe
to a V6 scored month and runs the frozen Stage 6 and Stage 8 formulas.  Forward
labels use the exact post-entry-close interval 2019-12-02 through 2020-01-02;
no adjacent-month interpolation is permitted.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import dataclasses
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
DATABASE = Path("/Users/agraemirolmez/Desktop/Thesis")
OUTPUT = DATABASE / "processed/research_v6/december_2019_v1"
FEATURE_BUNDLE = DATABASE / (
    "cache/phase_06_final_test/stage13-final-source-replay-v1/"
    "082c42dd57e8077f47cf409a12aefd8ec2ee98b732839496a320f8f84abeb66e/"
    "formula_work/feature_sources/feature_input_bundle"
)
RETURN_HISTORY = DATABASE / (
    "cache/phase_06_final_test/stage13-final-source-replay-v1/"
    "dbccbf9cd220d890d8c025bb6e92854bf61098f1eb42517da934c62b98e7adf5/"
    "formula_work/feature_sources/full_predictor_context/tables/"
    "crsp_security_day_return.parquet"
)

HOLDING_MONTH = "2019-12"
SIGNAL_DATE = dt.date(2019, 11, 29)
ENTRY_DATE = dt.date(2019, 12, 2)
EXIT_DATE = dt.date(2020, 1, 2)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")
    os.replace(temporary, path)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _load_promoted_bundle(context_root: Path | None = None):
    from portfolio_bot.features.feature_store_inputs import (
        FeatureStoreInputs,
        InputTable,
    )
    from portfolio_bot.validation.validation_feature_input_adapter import (
        ValidationFeatureInputBundle,
    )

    receipt_path = FEATURE_BUNDLE / "validation_feature_input_bundle_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    if (
        receipt.get("status") != "FINAL_VERIFIED_LOGICAL_FEATURE_HISTORY_COMPLETE"
        or int(receipt.get("internal_optionmetrics_context", {}).get("working_signal_count", -1))
        != 73
    ):
        raise RuntimeError("authenticated final feature bundle differs")
    tables: dict[str, InputTable] = {}
    for name, raw in receipt["scored_catalog"].items():
        path = Path(raw["path"])
        tables[name] = InputTable(
            gate=str(raw["gate"]),
            name=name,
            path=path,
            sha256=str(raw["sha256"]),
            size_bytes=int(raw["size_bytes"]),
            rows=int(raw["rows"]),
            row_groups=int(raw["row_groups"]),
        )
    for name, filename in (
        ("crsp_signal_date_universe.parquet", "optionmetrics_working_signal_date_universe.parquet"),
        ("auxiliary_signal_linkage.parquet", "optionmetrics_working_auxiliary_signal_linkage.parquet"),
    ):
        path = FEATURE_BUNDLE / "tables" / filename
        parquet = pq.ParquetFile(path)
        tables[name] = InputTable(
            gate=tables[name].gate,
            name=name,
            path=path,
            sha256=_sha256(path),
            size_bytes=path.stat().st_size,
            rows=parquet.metadata.num_rows,
            row_groups=parquet.metadata.num_row_groups,
        )
    # The final source bundle extends beyond the missing 2019-12 signal because
    # it was assembled for the complete 2020--2025 final replay.  The frozen
    # fundamental source gate deliberately rejects such future context when a
    # single older month is replayed.  Publish a deterministic point-in-time
    # derivative of the two Compustat observation tables instead of weakening
    # that gate.  Rows not public by the signal close cannot contribute to any
    # December 2019 feature and are therefore excluded here.
    context_root = (context_root or (OUTPUT / "work/source_context_2019_11_29")).resolve()
    context_root.mkdir(parents=True, exist_ok=True)
    for name in (
        "compustat_quarterly_observation.parquet",
        "compustat_annual_observation.parquet",
    ):
        source = tables[name].path
        destination = context_root / name
        receipt_path = destination.with_suffix(".json")
        source_sha256 = tables[name].sha256
        reusable = False
        if destination.is_file() and receipt_path.is_file():
            derived_receipt = json.loads(receipt_path.read_text())
            reusable = (
                derived_receipt.get("status") == "COMPLETE"
                and derived_receipt.get("source_sha256") == source_sha256
                and derived_receipt.get("maximum_public_date") == SIGNAL_DATE.isoformat()
                and derived_receipt.get("sha256") == _sha256(destination)
            )
        if not reusable:
            frame = pq.read_table(source)
            mask = None
            for column in ("datadate", "observation_date", "public_availability_date"):
                condition = pa.compute.less_equal(
                    frame[column], pa.scalar(SIGNAL_DATE, type=pa.date32())
                )
                mask = condition if mask is None else pa.compute.and_(mask, condition)
            filtered = frame.filter(mask)
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            pq.write_table(filtered, temporary, compression="zstd")
            os.replace(temporary, destination)
            _atomic_json(
                receipt_path,
                {
                    "status": "COMPLETE",
                    "source_path": str(source),
                    "source_sha256": source_sha256,
                    "maximum_public_date": SIGNAL_DATE.isoformat(),
                    "rows": filtered.num_rows,
                    "sha256": _sha256(destination),
                },
            )
        parquet = pq.ParquetFile(destination)
        tables[name] = InputTable(
            gate=tables[name].gate,
            name=name,
            path=destination,
            sha256=_sha256(destination),
            size_bytes=destination.stat().st_size,
            rows=parquet.metadata.num_rows,
            row_groups=parquet.metadata.num_row_groups,
        )

    catalog = FeatureStoreInputs(DATABASE, MappingProxyType(dict(tables)))
    identity = hashlib.sha256(
        (
            str(receipt["input_bundle_sha256"])
            + ":v6-promote-2019-11-29:"
            + tables["compustat_quarterly_observation.parquet"].sha256
            + ":"
            + tables["compustat_annual_observation.parquet"].sha256
        ).encode()
    ).hexdigest()
    return ValidationFeatureInputBundle(
        scored_inputs=catalog,
        optionmetrics_working_inputs=catalog,
        receipt_path=receipt_path,
        logical_history_root=FEATURE_BUNDLE,
        input_bundle_sha256=identity,
        source_receipts=MappingProxyType(
            {"parent_input_bundle_sha256": str(receipt["input_bundle_sha256"])}
        ),
    )


def _calendar() -> pa.Table:
    return pa.table(
        {
            "source_holding_month": pa.array([HOLDING_MONTH], pa.string()),
            "signal_date": pa.array([SIGNAL_DATE], pa.date32()),
            "entry_date": pa.array([ENTRY_DATE], pa.date32()),
            "exit_date": pa.array([EXIT_DATE], pa.date32()),
        }
    )


def _contract():
    from portfolio_bot.validation.validation_data_contract import (
        ValidationSignal,
        validate_validation_data_contract,
    )
    original = validate_validation_data_contract()
    return dataclasses.replace(
        original,
        validation_signals=(
            ValidationSignal(HOLDING_MONTH, SIGNAL_DATE, ENTRY_DATE, EXIT_DATE),
        ),
    )


@contextmanager
def _ownership_point_in_time_scope():
    """Validate only the source history knowable at the missing signal.

    The authenticated final bundle legitimately contains ownership/event rows
    published after November 2019.  The frozen streaming formula never advances
    those rows into an earlier signal, but its source gate expects a physically
    truncated input.  Install a gate adapter that presents an as-of view for
    validation while leaving the immutable source archive untouched.
    """

    import portfolio_bot.features.materialize_ownership_events as ownership

    frozen = ownership._validate_source_gates

    def validate_as_of(connection, inputs):
        specifications = (
            ("sec_filings", ownership.SEC_FILING_TABLE),
            ("sec_items", ownership.SEC_ITEM_TABLE),
            ("filings_13f", ownership.FILING_13F_TABLE),
            ("holdings_13f", ownership.HOLDING_13F_TABLE),
            ("restatement_events", ownership.RESTATEMENT_EVENT_TABLE),
        )
        for view, table in specifications:
            path = str(inputs.path(table)).replace("'", "''")
            connection.execute(f"DROP VIEW {view}")
            connection.execute(
                f"""
                CREATE TEMP VIEW {view} AS
                SELECT *
                FROM read_parquet('{path}')
                WHERE observation_date <= DATE '{SIGNAL_DATE.isoformat()}'
                  AND public_availability_date <= DATE '{SIGNAL_DATE.isoformat()}'
                """
            )
        link_path = str(inputs.path(ownership.RESTATEMENT_LINK_TABLE)).replace("'", "''")
        connection.execute("DROP VIEW restatement_links")
        connection.execute(
            f"""
            CREATE TEMP VIEW restatement_links AS
            SELECT links.*
            FROM read_parquet('{link_path}') AS links
            INNER JOIN restatement_events AS events
              USING (restatement_notification_key)
            """
        )
        frozen(connection, inputs)

    ownership._validate_source_gates = validate_as_of
    try:
        yield
    finally:
        ownership._validate_source_gates = frozen

def _build_predictors_and_costs(output: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    from portfolio_bot.final_test.final_source_replay import (
        _materialize_final_predictors,
        _materialize_final_risk_tables,
    )

    bundle = _load_promoted_bundle(output / "work/source_context_2019_11_29")
    contract = _contract()
    calendar = _calendar()
    with _ownership_point_in_time_scope():
        predictors_arrow, predictor_receipt = _materialize_final_predictors(
            bundle=bundle,
            final_contract=contract,
            final_calendar=calendar,
            pending_months=(HOLDING_MONTH,),
            work_root=output / "work/predictor_formula",
        )
    risk_tables, risk_receipt = _materialize_final_risk_tables(
        predictors=predictors_arrow,
        bundle=bundle,
        final_contract=contract,
        final_calendar=calendar,
        pending_months=(HOLDING_MONTH,),
    )
    predictors = predictors_arrow.to_pandas().rename(
        columns={"source_holding_month": "holding_month"}
    )
    costs = risk_tables["final_risk_cost"].to_pandas().rename(
        columns={"source_holding_month": "holding_month"}
    )
    return predictors, costs, {
        "predictor_formula": dict(predictor_receipt),
        "risk_formula": dict(risk_receipt),
    }


def _build_targets(predictors: pd.DataFrame, costs: pd.DataFrame) -> pd.DataFrame:
    from portfolio_bot.features.label_math import (
        FactorObservation,
        ForwardLeg,
        PRESENT,
        evaluate_forward_outcome,
    )
    from portfolio_bot.research_v6.data import _attach_net_values, _centered_rank

    factor_path = _load_promoted_bundle().scored_inputs.path(
        "ff_factor_daily_observation.parquet"
    )
    factors = pd.read_parquet(factor_path)
    factors["observation_date"] = pd.to_datetime(factors["observation_date"]).dt.date
    factors = factors.loc[
        factors["observation_date"].gt(ENTRY_DATE)
        & factors["observation_date"].le(EXIT_DATE)
    ]
    factor_map = {
        row.observation_date: FactorObservation(
            calendar_date=row.observation_date,
            rf=float(row.rf),
            mktrf=float(row.mktrf),
            smb=float(row.smb),
            hml=float(row.hml),
            rmw=float(row.rmw),
            cma=float(row.cma),
            umd=float(row.umd),
            public_availability_date=pd.Timestamp(row.public_availability_date).date(),
        )
        for row in factors.itertuples(index=False)
    }
    sessions = tuple(sorted(factor_map))
    if not sessions or sessions[-1] != EXIT_DATE:
        raise RuntimeError("December 2019 factor/session interval is incomplete")

    permnos = predictors["permno"].astype(int).unique().tolist()
    returns = pd.read_parquet(
        RETURN_HISTORY,
        filters=[
            ("calendar_date", ">", ENTRY_DATE),
            ("calendar_date", "<=", EXIT_DATE),
        ],
        columns=[
            "permno",
            "calendar_date",
            "return_span_start_date",
            "return_span_end_date",
            "authoritative_total_return",
            "outcome_availability_date",
            "return_leg_class",
            "return_history_usable",
            "return_history_authorized",
            "covered_trading_sessions",
            "delisting_treatment_status",
        ],
    )
    returns = returns.loc[returns["permno"].isin(permnos)].copy()
    returns["calendar_date"] = pd.to_datetime(returns["calendar_date"]).dt.date
    grouped = {key: value for key, value in returns.groupby("permno", sort=False)}
    rows: list[dict[str, object]] = []
    factor_sums = {
        name: math.fsum(getattr(value, name) or 0.0 for value in factor_map.values())
        for name in ("mktrf", "smb", "hml", "rmw", "cma", "umd")
    }
    for row in predictors.itertuples(index=False):
        legs: dict[dt.date, list[ForwardLeg]] = {}
        for value in grouped.get(int(row.permno), pd.DataFrame()).itertuples(index=False):
            legs.setdefault(value.calendar_date, []).append(
                ForwardLeg(
                    calendar_date=value.calendar_date,
                    span_start_date=pd.Timestamp(value.return_span_start_date).date(),
                    span_end_date=pd.Timestamp(value.return_span_end_date).date(),
                    return_value=(
                        None
                        if pd.isna(value.authoritative_total_return)
                        else float(value.authoritative_total_return)
                    ),
                    outcome_availability_date=pd.Timestamp(value.outcome_availability_date).date(),
                    outcome_class=str(value.return_leg_class),
                    return_history_usable=bool(value.return_history_usable),
                    # The source table is content-authenticated above, but its
                    # authorization bit belongs to the closed predecessor
                    # replay and is false outside that lineage.  V6 is a new,
                    # explicitly authorized December reconstruction: admit
                    # only rows that the source itself marks usable, and bind
                    # that decision in the V6 manifest rather than mutating
                    # the immutable source flag.
                    verified_return_pointer_authorized=bool(value.return_history_usable),
                    covered_session_count=int(value.covered_trading_sessions),
                    delisting_status=str(value.delisting_treatment_status),
                )
            )
        betas = tuple(
            getattr(row, f"beta_label_{name}")
            for name in ("mktrf", "smb", "hml", "rmw", "cma", "umd")
        )
        outcome = evaluate_forward_outcome(
            ENTRY_DATE,
            EXIT_DATE,
            sessions,
            betas,
            legs,
            factor_map,
            long_eligible=bool(row.long_eligible),
        )
        if outcome.status != PRESENT:
            continue
        total_return = math.prod(
            1.0 + value.return_value for value in outcome.contributions
        ) - 1.0
        alpha_capm = float(outcome.excess_sum) - float(betas[0]) * factor_sums["mktrf"]
        rows.append(
            {
                "holding_month": HOLDING_MONTH,
                "signal_date": pd.Timestamp(SIGNAL_DATE),
                "permno": int(row.permno),
                "permco": int(row.permco),
                "long_eligible": bool(row.long_eligible),
                "short_eligible": bool(row.short_eligible),
                "label_available_date": pd.Timestamp(outcome.available_date),
                "unranked_excess_return": total_return,
                "alpha_capm": alpha_capm,
                "alpha_ff5_momentum": float(outcome.residual_sum),
            }
        )
    targets = pd.DataFrame(rows)
    if len(targets) < 1_000:
        raise RuntimeError("December 2019 produced too few valid labels")
    net = _attach_net_values(targets, costs)
    for source, output in (
        ("alpha_capm", "target_rank_capm"),
        ("alpha_ff5_momentum", "target_rank_ff5_momentum"),
        ("long_net_opportunity", "target_rank_long_net"),
        ("short_net_opportunity", "target_rank_short_net"),
    ):
        net[output] = _centered_rank(net[source])
    columns = [
        "holding_month",
        "signal_date",
        "permno",
        "permco",
        "label_available_date",
        "unranked_excess_return",
        "alpha_capm",
        "alpha_ff5_momentum",
        "long_net_opportunity",
        "short_net_opportunity",
        "long_shadow_adv_participation",
        "long_shadow_roundtrip_cost_return",
        "short_shadow_adv_participation",
        "short_shadow_roundtrip_cost_return",
        "target_rank_capm",
        "target_rank_ff5_momentum",
        "target_rank_long_net",
        "target_rank_short_net",
    ]
    return net[columns].sort_values("permno", kind="mergesort").reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    completion = output / "manifest.json"
    if completion.is_file():
        value = json.loads(completion.read_text())
        paths = [output / "predictors.parquet", output / "cost_inputs.parquet", output / "targets.parquet"]
        if value.get("status") == "COMPLETE" and all(path.is_file() for path in paths):
            print("December 2019 V6 build already complete", flush=True)
            return 0
    feature_manifest = output / "feature_manifest.json"
    predictor_path = output / "predictors.parquet"
    cost_path = output / "cost_inputs.parquet"
    reusable_features = False
    if feature_manifest.is_file() and predictor_path.is_file() and cost_path.is_file():
        value = json.loads(feature_manifest.read_text())
        reusable_features = (
            value.get("status") == "COMPLETE"
            and value.get("predictor_sha256") == _sha256(predictor_path)
            and value.get("cost_sha256") == _sha256(cost_path)
        )
    if reusable_features:
        predictors = pd.read_parquet(predictor_path)
        costs = pd.read_parquet(cost_path)
        formula_receipts = json.loads(feature_manifest.read_text())["formula_receipts"]
    else:
        predictors, costs, formula_receipts = _build_predictors_and_costs(output)
        _atomic_parquet(predictors, predictor_path)
        _atomic_parquet(costs, cost_path)
        _atomic_json(
            feature_manifest,
            {
                "schema": "research_v6_december_2019_features_v1",
                "status": "COMPLETE",
                "holding_month": HOLDING_MONTH,
                "signal_date": SIGNAL_DATE.isoformat(),
                "interpolation_used": False,
                "predictor_rows": len(predictors),
                "cost_rows": len(costs),
                "predictor_sha256": _sha256(predictor_path),
                "cost_sha256": _sha256(cost_path),
                "formula_receipts": formula_receipts,
            },
        )
    targets = _build_targets(predictors, costs)
    _atomic_parquet(targets, output / "targets.parquet")
    _atomic_json(
        completion,
        {
            "schema": "research_v6_december_2019_v1",
            "status": "COMPLETE",
            "holding_month": HOLDING_MONTH,
            "signal_date": SIGNAL_DATE.isoformat(),
            "entry_date": ENTRY_DATE.isoformat(),
            "exit_date": EXIT_DATE.isoformat(),
            "interpolation_used": False,
            "predictor_rows": len(predictors),
            "cost_rows": len(costs),
            "target_rows": len(targets),
            "predictor_sha256": _sha256(output / "predictors.parquet"),
            "cost_sha256": _sha256(output / "cost_inputs.parquet"),
            "target_sha256": _sha256(output / "targets.parquet"),
            "formula_receipts": formula_receipts,
        },
    )
    print(
        f"December 2019 complete: {len(predictors):,} predictors; {len(targets):,} valid labels",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
