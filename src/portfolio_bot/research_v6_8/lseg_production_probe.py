"""Checkpointed LSEG capability and frozen-source replay probe for V6.8.

The probe is deliberately narrower than the production provider-parity gate.
It establishes that the signed-in Workspace session can retrieve the required
content families and that historical LSEG analyst data can still be replayed
from the exact signal date.  CRSP, Compustat, and OptionMetrics replacement
parity remains a separate downstream gate.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class LsegProductionProbeError(RuntimeError):
    """The LSEG session or replay evidence failed a required probe check."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def normalize_lseg_frame(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Apply the exact column naming convention used by the V6 downloader."""

    if frame.empty:
        return pd.DataFrame(columns=["cusip9"])
    result = frame.copy()
    instrument = next(
        (
            column
            for column in result.columns
            if str(column).strip().upper() == "INSTRUMENT"
        ),
        result.columns[0],
    )
    result = result.rename(columns={instrument: "cusip9"})
    result["cusip9"] = result["cusip9"].astype("string").str.strip().str.upper()
    names: list[str] = []
    seen: dict[str, int] = {}
    for column in result.columns:
        if column == "cusip9":
            names.append(column)
            continue
        base = str(column).strip().lower()
        for old, new in (
            (".", "_"),
            ("(", "_"),
            (")", ""),
            ("=", "_"),
            (",", "_"),
            (" ", "_"),
        ):
            base = base.replace(old, new)
        name = f"{prefix}__{base}"
        seen[name] = seen.get(name, 0) + 1
        names.append(name if seen[name] == 1 else f"{name}_{seen[name]}")
    result.columns = names
    return result.drop_duplicates("cusip9", keep="last")


def compare_replay(
    archive: pd.DataFrame,
    requery: pd.DataFrame,
    *,
    key: str = "cusip9",
    relative_tolerance: float = 1e-8,
    absolute_tolerance: float = 1e-10,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compare point-in-time replay values without treating missing as zero."""

    if archive[key].duplicated().any() or requery[key].duplicated().any():
        raise LsegProductionProbeError("replay key is not unique")
    fields = sorted((set(archive.columns) & set(requery.columns)) - {key})
    merged = archive[[key, *fields]].merge(
        requery[[key, *fields]], on=key, how="outer", suffixes=("_archive", "_requery")
    )
    rows: list[dict[str, Any]] = []
    for field in fields:
        left = merged[f"{field}_archive"]
        right = merged[f"{field}_requery"]
        is_date = "date" in field.lower()
        if is_date:
            left_value = pd.to_datetime(left, errors="coerce")
            right_value = pd.to_datetime(right, errors="coerce")
            both = left_value.notna() & right_value.notna()
            exact = (
                float((left_value[both] == right_value[both]).mean())
                if both.any()
                else None
            )
        else:
            left_value = pd.to_numeric(left, errors="coerce")
            right_value = pd.to_numeric(right, errors="coerce")
            both = left_value.notna() & right_value.notna()
            tolerance = absolute_tolerance + relative_tolerance * left_value[both].abs()
            exact = (
                float(((left_value[both] - right_value[both]).abs() <= tolerance).mean())
                if both.any()
                else None
            )
        rows.append(
            {
                "field": field,
                "kind": "date" if is_date else "numeric",
                "archive_nonmissing": int(left_value.notna().sum()),
                "requery_nonmissing": int(right_value.notna().sum()),
                "both_nonmissing": int(both.sum()),
                "absolute_missingness_difference": float(
                    abs(left_value.notna().mean() - right_value.notna().mean())
                ),
                "exact_share_when_comparable": exact,
            }
        )
    detail = pd.DataFrame(rows)
    economic = detail.loc[
        detail["kind"].eq("numeric")
        & detail["exact_share_when_comparable"].notna()
        & detail["both_nonmissing"].gt(0)
    ]
    weighted_exact = (
        float(
            np.average(
                economic["exact_share_when_comparable"],
                weights=economic["both_nonmissing"],
            )
        )
        if not economic.empty
        else None
    )
    summary = {
        "archive_rows": int(len(archive)),
        "requery_rows": int(len(requery)),
        "joined_rows": int(len(merged)),
        "compared_fields": int(len(detail)),
        "numeric_fields_with_overlap": int(len(economic)),
        "weighted_numeric_exact_share": weighted_exact,
        "maximum_missingness_difference": (
            float(detail["absolute_missingness_difference"].max())
            if not detail.empty
            else None
        ),
    }
    return detail, summary


def _request_with_retry(
    ld: Any,
    instruments: list[str],
    fields: list[str],
    parameters: dict[str, str] | None = None,
) -> pd.DataFrame:
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            kwargs: dict[str, Any] = {
                "universe": instruments,
                "fields": fields,
                "header_type": ld.HeaderType.NAME,
            }
            if parameters:
                kwargs["parameters"] = parameters
            return ld.get_data(**kwargs)
        except Exception as error:  # pragma: no cover - live provider path
            last_error = error
            if attempt < 3:
                time.sleep(2**attempt)
    raise LsegProductionProbeError(f"LSEG request failed: {last_error}")


def _dataframe_evidence(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "returned_fields": [str(column) for column in frame.columns],
        "rows_with_any_value": int(frame.dropna(how="all").shape[0]),
    }


def _lseg_identifier_text(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and np.isfinite(value):
        return str(int(value)) if float(value).is_integer() else str(value)
    text = str(value).strip()
    return text or None


def _market_bridge_metrics(
    reference: pd.DataFrame,
    bridge: pd.DataFrame,
    features: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    merged = reference[["permno", *features]].merge(
        bridge[["permno", *features]],
        on="permno",
        how="outer",
        suffixes=("_reference", "_bridge"),
        validate="one_to_one",
    )
    rows: list[dict[str, Any]] = []
    for feature in features:
        reference_value = pd.to_numeric(
            merged[f"{feature}_reference"], errors="coerce"
        )
        bridge_value = pd.to_numeric(merged[f"{feature}_bridge"], errors="coerce")
        comparable = reference_value.notna() & bridge_value.notna()
        if comparable.sum() >= 3:
            reference_rank = reference_value[comparable].rank(method="average", pct=True)
            bridge_rank = bridge_value[comparable].rank(method="average", pct=True)
            rank_correlation = float(reference_rank.corr(bridge_rank))
            median_rank_difference = float(
                (reference_rank - bridge_rank).abs().median()
            )
            relative_difference = (
                (bridge_value[comparable] - reference_value[comparable]).abs()
                / (reference_value[comparable].abs() + 1e-12)
            )
            median_relative_difference = float(relative_difference.median())
        else:
            rank_correlation = None
            median_rank_difference = None
            median_relative_difference = None
        missingness_difference = float(
            abs(reference_value.notna().mean() - bridge_value.notna().mean())
        )
        passed = bool(
            rank_correlation is not None
            and rank_correlation >= 0.95
            and median_rank_difference is not None
            and median_rank_difference <= 0.05
            and missingness_difference <= 0.02
        )
        rows.append(
            {
                "feature": feature,
                "comparable_rows": int(comparable.sum()),
                "spearman_rank_correlation": rank_correlation,
                "median_absolute_rank_difference": median_rank_difference,
                "median_absolute_relative_difference": median_relative_difference,
                "absolute_missingness_difference": missingness_difference,
                "pilot_gate_pass": passed,
            }
        )
    detail = pd.DataFrame(rows)
    return detail, {
        "compared_features": int(len(detail)),
        "features_passing_pilot_gate": int(detail["pilot_gate_pass"].sum()),
        "minimum_rank_correlation": float(detail["spearman_rank_correlation"].min()),
        "maximum_median_rank_difference": float(
            detail["median_absolute_rank_difference"].max()
        ),
        "maximum_missingness_difference": float(
            detail["absolute_missingness_difference"].max()
        ),
    }


def _consolidated_ric_history(
    ld: Any,
    *,
    ticker: str,
    cusip9: str | None = None,
    expected_signal_price: float,
    start: str,
    end: str,
) -> tuple[str, pd.DataFrame] | None:
    """Resolve the US consolidated RIC, never a venue-only quote RIC."""

    discoveries: list[pd.DataFrame] = []
    # CUSIP is the authoritative security-level query.  A broad ticker such as
    # BIO can otherwise rank unrelated companies ahead of the exact US share
    # class.  Ticker remains a fallback for rows without a usable CUSIP.
    queries = list(dict.fromkeys([cusip9, ticker]))
    for query in (value for value in queries if value):
        last_error: Exception | None = None
        for attempt in range(4):
            try:
                discoveries.append(
                    ld.discovery.search(
                        query=query,
                        view=ld.discovery.Views.EQUITY_QUOTES,
                        top=20,
                    )
                )
                last_error = None
                break
            except Exception as error:  # pragma: no cover - live provider path
                last_error = error
                if attempt < 3:
                    time.sleep(2**attempt)
        if last_error is not None:
            raise LsegProductionProbeError(
                f"LSEG equity discovery failed for {query}: {last_error}"
            )
    discovery = (
        pd.concat(discoveries, ignore_index=True).drop_duplicates("RIC")
        if discoveries
        else pd.DataFrame()
    )
    if discovery.empty or not {"DocumentTitle", "RIC"}.issubset(discovery.columns):
        return None
    candidates = discovery.loc[
        discovery["DocumentTitle"].astype(str).str.contains(
            "Consolidated", case=False, na=False
        )
    ]
    best: tuple[float, str, pd.DataFrame, float] | None = None
    for _, candidate_row in candidates.head(4).iterrows():
        ric = str(candidate_row["RIC"])
        history_identifiers = [ric]
        permid = _lseg_identifier_text(candidate_row.get("PermID"))
        if permid is not None and permid not in history_identifiers:
            history_identifiers.append(permid)
        history = pd.DataFrame()
        history_identifier: str | None = None
        for identifier in history_identifiers:
            try:
                candidate_history = ld.get_history(
                    universe=[identifier],
                    fields=[
                        "TR.PriceClose",
                        "TR.TotalReturn",
                        "TR.AccumulatedVolume",
                    ],
                    interval="1D",
                    start=start,
                    end=end,
                )
            except Exception:  # pragma: no cover - live provider path
                continue
            required = {"Price Close", "Total Return", "Accumulated Volume"}
            if candidate_history.empty or not required.issubset(
                candidate_history.columns
            ):
                continue
            candidate_history = candidate_history.dropna(
                subset=list(required)
            ).copy()
            if len(candidate_history) < 5:
                continue
            history = candidate_history
            history_identifier = identifier
            break
        if history.empty or history_identifier is None:
            continue
        observed_price = float(history["Price Close"].iloc[-1])
        try:
            raw_history = ld.get_history(
                universe=[history_identifier],
                fields=["TRDPRC_1"],
                interval="1D",
                start=(pd.Timestamp(end) - timedelta(days=10)).date().isoformat(),
                end=end,
                adjustments="unadjusted",
            )
            if not raw_history.empty and "TRDPRC_1" in raw_history.columns:
                raw_price = pd.to_numeric(
                    raw_history["TRDPRC_1"], errors="coerce"
                ).dropna()
                if not raw_price.empty:
                    observed_price = float(raw_price.iloc[-1])
        except Exception:  # pragma: no cover - live provider path
            pass
        relative_difference = abs(observed_price - expected_signal_price) / max(
            abs(expected_signal_price), 1e-12
        )
        history.attrs["lseg_history_identifier"] = history_identifier
        candidate = (relative_difference, ric, history, observed_price)
        if best is None or candidate[0] < best[0]:
            best = candidate
    if best is None or best[0] > 0.02:
        return None
    best[2].attrs["unadjusted_signal_price"] = best[3]
    return best[1], best[2]


def _write_progress(output: Path, completed: list[str], current: str) -> None:
    _atomic_json(
        output / "progress.json",
        {
            "schema": "research_v6_8_lseg_production_probe_progress_v1",
            "status": "RUNNING",
            "updated_at_utc": _utc_now(),
            "completed_stages": completed,
            "current_stage": current,
        },
    )


def run_lseg_production_probe(
    *,
    output: Path,
    archive_month: Path,
    archive_contract: Path,
    model_feature_month: Path | None = None,
    sample_size: int = 12,
    replay_prefixes: tuple[str, ...] = ("fy1", "signals"),
) -> dict[str, Any]:
    """Run the live entitlement probe and one exact historical-source replay."""

    try:
        import lseg.data as ld
    except ImportError as error:  # pragma: no cover - environment-specific
        raise LsegProductionProbeError(
            "run this probe with /opt/anaconda3/envs/finance/bin/python"
        ) from error

    output.mkdir(parents=True, exist_ok=True)
    completed: list[str] = []
    _write_progress(output, completed, "load_authenticated_archive")
    archive = pd.read_parquet(archive_month).sort_values("market_cap_usd").reset_index(drop=True)
    if len(archive) < sample_size:
        raise LsegProductionProbeError("archive month is smaller than the requested sample")
    sample_indices = np.linspace(0, len(archive) - 1, sample_size).round().astype(int)
    sample = archive.iloc[sample_indices].drop_duplicates("cusip9").copy()
    source_contract = json.loads(archive_contract.read_text(encoding="utf-8"))
    signal_dates = sample["signal_date"].astype(str).unique().tolist()
    if len(signal_dates) != 1:
        raise LsegProductionProbeError("sample does not have one exact signal date")
    signal_date = signal_dates[0]
    completed.append("load_authenticated_archive")

    evidence: dict[str, Any] = {}
    details: list[pd.DataFrame] = []
    requery_parts: list[pd.DataFrame] = []
    _write_progress(output, completed, "open_workspace_session")
    ld.open_session()
    completed.append("open_workspace_session")
    try:
        _write_progress(output, completed, "same_source_historical_replay")
        for prefix in replay_prefixes:
            contract_key = prefix if prefix in source_contract["fields"] else prefix.upper()
            fields = source_contract["fields"][contract_key]
            response = _request_with_retry(
                ld,
                sample["cusip9"].astype(str).tolist(),
                fields,
                {"SDate": signal_date, "EDate": signal_date, "Frq": "D"},
            )
            requery_parts.append(normalize_lseg_frame(response, prefix))
        requery = requery_parts[0]
        for part in requery_parts[1:]:
            requery = requery.merge(part, on="cusip9", how="outer", validate="one_to_one")
        replay_fields = [
            column
            for column in requery.columns
            if column != "cusip9" and column in sample.columns
        ]
        replay_detail, replay_summary = compare_replay(
            sample[["cusip9", *replay_fields]],
            requery[["cusip9", *replay_fields]],
        )
        replay_detail.to_csv(output / "same_source_replay_fields.csv", index=False)
        evidence["same_source_historical_replay"] = {
            **replay_summary,
            "holding_month": str(sample["holding_month"].iloc[0]),
            "signal_date": signal_date,
            "sample_size": int(len(sample)),
            "sample_selection": "equally spaced observations after ascending market-cap sort",
        }
        completed.append("same_source_historical_replay")

        representatives = ["AAPL.O", "MSFT.O", "NVDA.O", "JPM.N", "XOM.N", "F.N"]
        _write_progress(output, completed, "fundamentals_estimates_identity")
        current_groups = {
            "analyst_cam": [
                "TR.EPSMean(Period=FY1)",
                "TR.RevenueMean(Period=FY1)",
                "TR.RecMean",
                "TR.PriceTargetMean",
                "TR.CombinedAlphaCountryRank",
            ],
            "fundamentals": [
                "TR.TotalAssetsReported",
                "TR.TotalEquity",
                "TR.Revenue",
                "TR.NetIncome",
                "TR.TotalDebt",
                "TR.CompanyMarketCap",
            ],
            "identity": [
                "TR.ExchangeName",
                "TR.SICIndustryCode",
                "TR.HeadquartersCountry",
            ],
            "historical_volatility": [
                "TR.Volatility30D",
                "TR.Volatility60D",
                "TR.Volatility90D",
                "TR.Volatility180D",
            ],
        }
        for name, fields in current_groups.items():
            response = _request_with_retry(ld, representatives, fields)
            evidence[name] = _dataframe_evidence(response)
        completed.append("fundamentals_estimates_identity")

        _write_progress(output, completed, "daily_price_volume_history")
        history_start = date.today() - timedelta(days=50)
        history_end = date.today() - timedelta(days=1)
        price_history = ld.get_history(
            universe=representatives[:3],
            fields=["TRDPRC_1"],
            interval="1D",
            start=history_start.isoformat(),
            end=history_end.isoformat(),
        )
        volume_history = ld.get_history(
            universe=representatives[:3],
            fields=["ACVOL_UNS"],
            interval="1D",
            start=history_start.isoformat(),
            end=history_end.isoformat(),
        )
        evidence["daily_price_history"] = _dataframe_evidence(price_history)
        evidence["daily_volume_history"] = _dataframe_evidence(volume_history)
        completed.append("daily_price_volume_history")

        if model_feature_month is not None:
            _write_progress(output, completed, "market_price_bridge_pilot")
            market_features = [
                "return_total_5d",
                "return_total_63d",
                "return_total_126d",
                "return_total_252d",
                "realized_volatility_21d",
                "realized_volatility_63d",
                "realized_volatility_126d",
                "realized_volatility_252d",
                "log_absolute_price",
                "median_dollar_volume_60d",
            ]
            model_features = pd.read_parquet(
                model_feature_month,
                columns=["permno", *market_features],
            )
            market_population = archive.merge(
                model_features, on="permno", how="inner", validate="one_to_one"
            )
            market_population = market_population.loc[
                market_population["ticker"].astype(str).str.fullmatch(r"[A-Z0-9]+")
                & market_population["log_absolute_price"].notna()
            ].sort_values("market_cap_usd").reset_index(drop=True)
            market_sample_size = min(20, len(market_population))
            market_indices = (
                np.linspace(0, len(market_population) - 1, market_sample_size)
                .round()
                .astype(int)
            )
            market_sample = market_population.iloc[market_indices].drop_duplicates(
                "permno"
            )
            bridge_rows: list[dict[str, Any]] = []
            mapping_rows: list[dict[str, Any]] = []
            for _, row in market_sample.iterrows():
                resolved = _consolidated_ric_history(
                    ld,
                    ticker=str(row["ticker"]),
                    expected_signal_price=float(np.exp(row["log_absolute_price"])),
                    start="2024-05-01",
                    end=signal_date,
                )
                if resolved is None:
                    mapping_rows.append(
                        {
                            "permno": int(row["permno"]),
                            "ticker": str(row["ticker"]),
                            "mapping_status": "UNRESOLVED",
                            "consolidated_ric": None,
                        }
                    )
                    continue
                ric, history = resolved
                total_return = pd.to_numeric(
                    history["Total Return"], errors="coerce"
                ) / 100.0
                price = pd.to_numeric(history["Price Close"], errors="coerce")
                volume = pd.to_numeric(
                    history["Accumulated Volume"], errors="coerce"
                )
                valid = total_return.notna() & price.notna() & volume.notna()
                total_return = total_return[valid]
                price = price[valid]
                volume = volume[valid]
                if len(total_return) < 252:
                    continue
                values: dict[str, Any] = {
                    "permno": int(row["permno"]),
                    "log_absolute_price": float(np.log(abs(price.iloc[-1]))),
                    "median_dollar_volume_60d": float(
                        np.median((price * volume).iloc[-60:])
                    ),
                }
                for window in (5, 63, 126, 252):
                    values[f"return_total_{window}d"] = float(
                        np.prod(1.0 + total_return.iloc[-window:]) - 1.0
                    )
                for window in (21, 63, 126, 252):
                    values[f"realized_volatility_{window}d"] = float(
                        total_return.iloc[-window:].std(ddof=1) * np.sqrt(252.0)
                    )
                bridge_rows.append(values)
                mapping_rows.append(
                    {
                        "permno": int(row["permno"]),
                        "ticker": str(row["ticker"]),
                        "mapping_status": "CONSOLIDATED_RIC_RESOLVED",
                        "consolidated_ric": ric,
                    }
                )
            bridge = pd.DataFrame(bridge_rows)
            mapping = pd.DataFrame(mapping_rows)
            if bridge.empty:
                raise LsegProductionProbeError(
                    "market bridge pilot resolved no consolidated histories"
                )
            market_detail, market_summary = _market_bridge_metrics(
                market_sample[["permno", *market_features]],
                bridge[["permno", *market_features]],
                market_features,
            )
            market_detail.to_csv(output / "market_bridge_features.csv", index=False)
            mapping.to_csv(output / "market_bridge_identifiers.csv", index=False)
            evidence["market_price_bridge_pilot"] = {
                **market_summary,
                "sample_size": int(len(market_sample)),
                "resolved_consolidated_rics": int(
                    mapping["mapping_status"].eq("CONSOLIDATED_RIC_RESOLVED").sum()
                ),
                "identifier_rule": "LSEG Equity Quotes discovery result explicitly labelled Consolidated; venue-only quote RICs are forbidden",
                "pilot_only": True,
            }
            completed.append("market_price_bridge_pilot")

        _write_progress(output, completed, "opra_options")
        option_contracts: list[str] = []
        option_underlyings = ["AAPL.O", "JPM.N", "F.N"]
        for underlying in option_underlyings:
            result = ld.discovery.search(
                view=ld.discovery.Views.EQUITY_DERIVATIVE_QUOTES,
                filter=(
                    f"UnderlyingQuoteRIC eq '{underlying}' and "
                    f"ExpiryDate gt {date.today().isoformat()}"
                ),
                top=1,
            )
            if not result.empty and "RIC" in result:
                option_contracts.append(str(result["RIC"].iloc[0]))
        if not option_contracts:
            raise LsegProductionProbeError("OPRA discovery returned no option contracts")
        option_snapshot = _request_with_retry(
            ld,
            option_contracts,
            [
                "BID",
                "ASK",
                "TRDPRC_1",
                "ACVOL_1",
                "IMP_VOLT",
                "TR.OpenInterest",
                "TR.AccumulatedVolume",
                "TR.ClosePrice",
                "TR.OPWCloseImpliedVolatility",
                "TR.OPWCloseDelta",
            ],
        )
        option_history = _request_with_retry(
            ld,
            [option_contracts[0]],
            [
                "TR.OpenInterest.Date",
                "TR.OpenInterest",
                "TR.AccumulatedVolume.Date",
                "TR.AccumulatedVolume",
                "TR.ClosePrice.Date",
                "TR.ClosePrice",
            ],
            {
                "SDate": (date.today() - timedelta(days=14)).isoformat(),
                "EDate": (date.today() - timedelta(days=1)).isoformat(),
                "Frq": "D",
            },
        )
        evidence["opra_discovery"] = {
            "underlyings_requested": len(option_underlyings),
            "underlyings_with_contract": len(option_contracts),
        }
        evidence["option_snapshot"] = _dataframe_evidence(option_snapshot)
        evidence["option_history"] = _dataframe_evidence(option_history)
        completed.append("opra_options")
    finally:
        ld.close_session()

    replay = evidence["same_source_historical_replay"]
    capability_names = [
        "analyst_cam",
        "fundamentals",
        "identity",
        "historical_volatility",
        "daily_price_history",
        "daily_volume_history",
        "option_snapshot",
        "option_history",
    ]
    capability_pass = all(evidence[name]["rows"] > 0 for name in capability_names)
    market_pilot_pass = bool(
        model_feature_month is None
        or (
            evidence["market_price_bridge_pilot"]["features_passing_pilot_gate"]
            == evidence["market_price_bridge_pilot"]["compared_features"]
            and evidence["market_price_bridge_pilot"]["resolved_consolidated_rics"]
            == evidence["market_price_bridge_pilot"]["sample_size"]
        )
    )
    replay_pass = bool(
        replay["weighted_numeric_exact_share"] is not None
        and replay["weighted_numeric_exact_share"] >= 0.999
        and replay["maximum_missingness_difference"] <= 0.02
    )
    receipt = {
        "schema": "research_v6_8_lseg_production_probe_v1",
        "created_at_utc": _utc_now(),
        "status": (
            "CAPABILITY_AND_SAME_SOURCE_REPLAY_PASS_PROVIDER_BRIDGE_PENDING"
            if capability_pass and replay_pass and market_pilot_pass
            else "PROBE_FAILED"
        ),
        "finance_runtime_required": True,
        "archive_month_sha256": _sha256(archive_month),
        "archive_contract_sha256": _sha256(archive_contract),
        "no_economic_values_persisted": True,
        "checks": {
            "capability_pass": capability_pass,
            "same_source_replay_pass": replay_pass,
            "market_price_bridge_pilot_pass": market_pilot_pass,
            "provider_replacement_parity_complete": False,
        },
        "evidence": evidence,
        "remaining_gates": [
            "full-window/full-universe CRSP-to-LSEG price, liquidity, universe, and identifier overlap parity beyond the passing pilot",
            "Compustat-to-LSEG fundamental definition, lag, and currency parity",
            "OptionMetrics-to-LSEG OPRA aggregate-feature parity",
            "causal signal-close factor policy",
            "current 318-column signal package and Alpaca dry run",
        ],
    }
    _atomic_json(output / "receipt.json", receipt)
    _atomic_json(
        output / "progress.json",
        {
            "schema": "research_v6_8_lseg_production_probe_progress_v1",
            "status": (
                "COMPLETE"
                if capability_pass and replay_pass and market_pilot_pass
                else "FAILED"
            ),
            "updated_at_utc": _utc_now(),
            "completed_stages": completed,
        },
    )
    if not capability_pass or not replay_pass or not market_pilot_pass:
        raise LsegProductionProbeError("one or more production capability checks failed")
    return receipt


__all__ = [
    "LsegProductionProbeError",
    "compare_replay",
    "normalize_lseg_frame",
    "run_lseg_production_probe",
]
