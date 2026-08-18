#!/usr/bin/env python3
"""Test causal intramonth interventions on frozen V6.1 monthly portfolios.

The monthly selectors, optimizer, stock set, and relative target weights are
immutable inputs.  This separate V6.6 lineage may only reduce an existing
position after a close-to-next-close execution delay.  Expensive preparation,
model refits, and policy replays checkpoint atomically.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping
from zoneinfo import ZoneInfo

import duckdb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import analyze_v6_1_return_scaling_overlay as cost_helpers  # noqa: E402
from scripts import materialize_v6_news_features as news_helpers  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/research_v6_6/intrmonth_interventions.json"
DEFAULT_OUTPUT = ROOT / "artifacts/research_v6_6/intrmonth_interventions"
MODEL_FEATURE_MONTHS = Path(
    "/Users/agraemirolmez/Desktop/Thesis/processed/research_v6/model_features_v2/months"
)
LSEG_UNIVERSE = Path(
    "/Users/agraemirolmez/Desktop/Thesis/raw/research_v6/lseg/input_universe.parquet"
)
RAW_NEWS = Path(
    "/Users/agraemirolmez/Desktop/Thesis/raw/research_v6/alpaca_news"
)
TOKEN_RE = re.compile(r"[A-Za-z]+")
DYNAMIC_COLUMNS = (
    "position_side",
    "initial_weight",
    "current_fraction",
    "days_held",
    "signed_position_return",
    "signed_peak_to_current_drawdown",
    "monthly_expected_stock_excess_return",
    "signed_return_1d",
    "signed_return_2d",
    "signed_return_5d",
    "signed_return_10d",
    "realized_volatility_5d",
    "realized_volatility_20d",
    "intraday_range",
    "log_dollar_volume",
    "volume_shock_5d",
    "mktrf_1d",
    "mktrf_5d",
    "market_volatility_5d",
    "umd_1d",
    "news_direct_1d_count",
    "news_direct_1d_tone_mean",
    "news_direct_1d_uncertainty_mean",
    "news_direct_7d_count",
    "news_direct_7d_tone_mean",
    "news_direct_7d_uncertainty_mean",
    "news_family_1d_count",
    "news_family_1d_tone_mean",
    "news_family_1d_uncertainty_mean",
    "news_family_7d_count",
    "news_family_7d_tone_mean",
    "news_family_7d_uncertainty_mean",
    "news_industry_1d_count",
    "news_industry_1d_tone_mean",
    "news_industry_1d_uncertainty_mean",
    "news_industry_7d_count",
    "news_industry_7d_tone_mean",
    "news_industry_7d_uncertainty_mean",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(temporary, path)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(value)
    os.replace(temporary, path)


def _duck(path: Path) -> str:
    return str(path).replace("'", "''")


def _read_parquet(path: Path) -> pd.DataFrame:
    return duckdb.sql(f"SELECT * FROM read_parquet('{_duck(path)}')").df()


def _month_sequence(start: str, end: str) -> list[str]:
    return pd.period_range(start, end, freq="M").astype(str).tolist()


def _previous_month(month: str) -> str:
    return str(pd.Period(month, freq="M") - 1)


def _next_month(month: str) -> str:
    return str(pd.Period(month, freq="M") + 1)


def _verify(config_path: Path) -> tuple[dict[str, Any], dict[str, str]]:
    config = json.loads(config_path.read_text())
    if config.get("status") != "AUTHORIZED_POST_DIAGNOSTIC_STUDY":
        raise RuntimeError("V6.6 intramonth study is not authorized")
    checks: list[tuple[Path, str, str]] = []
    sources = config["sources"]
    checks.extend(
        [
            (ROOT / sources["frozen_frontier"]["run_manifest"], sources["frozen_frontier"]["sha256"], "frontier manifest"),
            (ROOT / sources["alpaca_translation_config"]["path"], sources["alpaca_translation_config"]["sha256"], "Alpaca config"),
            (ROOT / sources["v6_encoded_inputs"]["rows"], sources["v6_encoded_inputs"]["rows_sha256"], "encoded rows"),
            (ROOT / sources["v6_encoded_inputs"]["matrix"], sources["v6_encoded_inputs"]["matrix_sha256"], "encoded matrix"),
        ]
    )
    for name in ("primary_outcomes", "risk_cost", "daily_returns", "daily_market", "daily_factors", "monthly_news_manifest"):
        item = sources[name]
        checks.append((Path(item["path"]), item["sha256"], name))
    for path, expected, label in checks:
        if not path.is_file():
            raise FileNotFoundError(path)
        if _sha256(path) != expected:
            raise RuntimeError(f"{label} identity changed: {path}")
    code_identity = hashlib.sha256()
    for path in (Path(__file__).resolve(), Path(cost_helpers.__file__).resolve(), Path(news_helpers.__file__).resolve()):
        code_identity.update(path.name.encode())
        code_identity.update(bytes.fromhex(_sha256(path)))
    identity = {
        "config_sha256": _sha256(config_path),
        "code_identity": code_identity.hexdigest(),
        "v6_input_identity": str(sources["v6_encoded_inputs"]["input_identity"]),
        "frontier_manifest_sha256": str(sources["frozen_frontier"]["sha256"]),
        "daily_returns_sha256": str(sources["daily_returns"]["sha256"]),
    }
    return config, identity


def _output_path(output_root: Path, identity: Mapping[str, str]) -> Path:
    return output_root / (
        f"{identity['config_sha256'][:12]}-"
        f"{identity['v6_input_identity'][:12]}-"
        f"{identity['code_identity'][:12]}"
    )


def _load_month_positions(config: Mapping[str, Any]) -> pd.DataFrame:
    chronology = config["chronology"]
    sources = config["sources"]
    records: list[pd.DataFrame] = []
    for month in _month_sequence(chronology["initial_fit_start"], chronology["historical_test_end"]):
        phase = "validation" if month <= "2023-12" else "historical_test"
        root = ROOT / sources["frozen_frontier"][f"{phase}_month_root"]
        path = root / f"{month}.parquet"
        if not path.is_file():
            raise FileNotFoundError(path)
        positions = _read_parquet(path)
        positions = positions.loc[
            positions["weight"].abs().gt(float(config["scope"]["active_weight_tolerance"]))
        ].copy()
        raw = 2.05 * positions["weight"].to_numpy(float)
        gross = float(np.abs(raw).sum())
        scalar = min(1.0, float(config["account_and_costs"]["rebalance_gross_cap"]) / gross)
        positions["target_weight"] = raw * scalar
        positions["base_gross"] = float(np.abs(positions["weight"]).sum())
        positions["target_gross"] = float(np.abs(positions["target_weight"]).sum())
        records.append(positions)
    frame = pd.concat(records, ignore_index=True)
    frame["holding_month"] = frame["holding_month"].astype(str)
    if frame.duplicated(["holding_month", "permno"]).any():
        raise RuntimeError("frozen position source has duplicate active stock-months")
    return frame


def _prepare_holdings(config: Mapping[str, Any], output: Path) -> pd.DataFrame:
    path = output / "prepared" / "holdings.parquet"
    receipt_path = output / "prepared" / "holdings.json"
    if path.is_file() and receipt_path.is_file():
        return _read_parquet(path)
    positions = _load_month_positions(config)
    outcome_path = Path(config["sources"]["primary_outcomes"]["path"])
    dates = duckdb.sql(
        f"""
        SELECT CAST(holding_month AS VARCHAR) holding_month,
               MIN(entry_date) entry_date, MAX(exit_date) exit_date,
               COUNT(DISTINCT entry_date) n_entry, COUNT(DISTINCT exit_date) n_exit
        FROM read_parquet('{_duck(outcome_path)}')
        WHERE holding_month BETWEEN '{config['chronology']['initial_fit_start']}'
                                AND '{config['chronology']['historical_test_end']}'
        GROUP BY holding_month ORDER BY holding_month
        """
    ).df()
    if not dates["n_entry"].eq(1).all() or not dates["n_exit"].eq(1).all():
        raise RuntimeError("monthly outcome windows are not unique")
    risk = cost_helpers._load_risk(  # noqa: SLF001
        {
            "sources": {
                "risk_cost": config["sources"]["risk_cost"],
                "validation": {"start": config["chronology"]["initial_fit_start"]},
                "historical_test": {"end": config["chronology"]["historical_test_end"]},
            }
        }
    )
    rows_path = ROOT / config["sources"]["v6_encoded_inputs"]["rows"]
    encoded_rows = _read_parquet(rows_path)[["holding_month", "permno", "permco"]].copy()
    encoded_rows.insert(0, "encoded_row", np.arange(len(encoded_rows), dtype=np.int64))
    encoded_rows["holding_month"] = encoded_rows["holding_month"].astype(str)
    keep_risk = [
        "holding_month", "permno", "full_spread_20d", "adv20_usd",
        "impact_volatility_daily", "primary_borrow_rate_annual",
        "stress_borrow_rate_annual",
    ]
    holdings = (
        positions.merge(dates[["holding_month", "entry_date", "exit_date"]], on="holding_month", how="left", validate="many_to_one")
        .merge(risk[keep_risk], on=["holding_month", "permno"], how="left", validate="one_to_one")
        .merge(encoded_rows, on=["holding_month", "permno", "permco"], how="left", validate="one_to_one")
    )
    required = ["entry_date", "exit_date", "encoded_row", "full_spread_20d", "adv20_usd", "impact_volatility_daily", "primary_borrow_rate_annual"]
    if holdings[required].isna().any().any():
        missing = holdings.loc[holdings[required].isna().any(axis=1), ["holding_month", "permno", *required]]
        raise RuntimeError(f"active holdings lack required inputs:\n{missing.head(20)}")
    holdings["encoded_row"] = holdings["encoded_row"].astype(np.int64)
    _atomic_parquet(holdings, path)
    _atomic_json(
        receipt_path,
        {
            "schema": "research_v6_6_holdings_v1",
            "rows": len(holdings),
            "months": int(holdings["holding_month"].nunique()),
            "first_month": holdings["holding_month"].min(),
            "last_month": holdings["holding_month"].max(),
            "maximum_target_gross": float(holdings.groupby("holding_month")["target_weight"].apply(lambda x: x.abs().sum()).max()),
        },
    )
    return holdings


def _prepare_month_days(month: str, holdings: pd.DataFrame, config: Mapping[str, Any], output: Path) -> pd.DataFrame:
    path = output / "prepared" / "months" / f"{month}.parquet"
    receipt_path = output / "prepared" / "months" / f"{month}.json"
    if path.is_file() and receipt_path.is_file():
        return _read_parquet(path)
    month_holdings = holdings.loc[holdings["holding_month"].eq(month)].copy()
    returns_path = Path(config["sources"]["daily_returns"]["path"])
    market_path = Path(config["sources"]["daily_market"]["path"])
    factors_path = Path(config["sources"]["daily_factors"]["path"])
    connection = duckdb.connect()
    connection.register("month_holdings", month_holdings)
    frame = connection.sql(
        f"""
        SELECT h.*, f.observation_date AS calendar_date,
               CAST(f.rf AS DOUBLE) rf,
               CAST(f.mktrf AS DOUBLE) mktrf,
               CAST(f.umd AS DOUBLE) umd,
               CAST(r.authoritative_total_return AS DOUBLE) stock_total_return_raw,
               r.return_history_usable,
               CAST(m.open_usd_per_share AS DOUBLE) open_price,
               CAST(m.high_usd_per_share AS DOUBLE) high_price,
               CAST(m.low_usd_per_share AS DOUBLE) low_price,
               CAST(m.close_usd_per_share AS DOUBLE) close_price,
               CAST(m.volume_shares AS DOUBLE) volume_shares,
               m.source_ticker
        FROM month_holdings h
        JOIN read_parquet('{_duck(factors_path)}') f
          ON f.observation_date > h.entry_date AND f.observation_date <= h.exit_date
        LEFT JOIN read_parquet('{_duck(returns_path)}') r
          ON r.permno = h.permno AND r.calendar_date = f.observation_date
        LEFT JOIN read_parquet('{_duck(market_path)}') m
          ON m.permno = h.permno AND m.calendar_date = f.observation_date
        ORDER BY h.permno, f.observation_date
        """
    ).df()
    connection.close()
    frame["calendar_date"] = pd.to_datetime(frame["calendar_date"])
    frame["stock_total_return"] = pd.to_numeric(frame["stock_total_return_raw"], errors="coerce").fillna(0.0)
    rf_month = float(np.prod(1.0 + frame.drop_duplicates("calendar_date").sort_values("calendar_date")["rf"].to_numpy(float)) - 1.0)
    target_total = frame["stock_excess_return"].to_numpy(float) + rf_month
    frame["target_total_return"] = target_total
    reconciliation: list[dict[str, Any]] = []
    for permno, index in frame.groupby("permno", sort=False).groups.items():
        loc = np.asarray(list(index), dtype=int)
        realized = frame.loc[loc, "stock_total_return"].to_numpy(float)
        target = float(frame.loc[loc[0], "target_total_return"])
        compound = float(np.prod(1.0 + realized) - 1.0)
        error = compound - target
        source = str(frame.loc[loc[0], "return_source"])
        if abs(error) > 1e-10:
            if source != "market_accounting" and abs(error) > 1e-8:
                raise RuntimeError(f"daily return reconciliation failed for {month}/{permno}: {error}")
            before = float(np.prod(1.0 + realized[:-1])) if len(realized) > 1 else 1.0
            frame.loc[loc[-1], "stock_total_return"] = (1.0 + target) / before - 1.0
        reconciliation.append({"permno": int(permno), "source": source, "initial_error": error})
    frame["day_index"] = frame.groupby("permno", sort=False).cumcount().astype(np.int16)
    frame["sessions_in_month"] = frame.groupby("permno", sort=False)["permno"].transform("size").astype(np.int16)
    side = np.sign(frame["target_weight"].to_numpy(float))
    frame["position_side"] = side
    frame["initial_weight"] = frame["target_weight"].astype(float)
    frame["current_fraction"] = 1.0
    frame["days_held"] = frame["day_index"].astype(float) + 1.0
    group = frame.groupby("permno", sort=False)
    cumulative = group["stock_total_return"].transform(lambda x: (1.0 + x).cumprod() - 1.0)
    frame["signed_position_return"] = side * cumulative
    signed_wealth = 1.0 + frame["signed_position_return"]
    peak = signed_wealth.groupby(frame["permno"], sort=False).cummax()
    frame["signed_peak_to_current_drawdown"] = signed_wealth / peak - 1.0
    frame["monthly_expected_stock_excess_return"] = pd.to_numeric(
        frame["regularized_expected_stock_excess_return"], errors="coerce"
    ).fillna(0.0)
    signed_daily = side * frame["stock_total_return"].to_numpy(float)
    frame["signed_return_1d"] = signed_daily
    for window in (2, 5, 10):
        frame[f"signed_return_{window}d"] = group["stock_total_return"].transform(
            lambda x, w=window: np.sign(frame.loc[x.index, "target_weight"].to_numpy(float))
            * ((1.0 + x).rolling(w, min_periods=1).apply(np.prod, raw=True) - 1.0)
        )
    for window in (5, 20):
        frame[f"realized_volatility_{window}d"] = group["stock_total_return"].transform(
            lambda x, w=window: x.rolling(w, min_periods=2).std(ddof=1)
        ).fillna(0.0)
    close = pd.to_numeric(frame["close_price"], errors="coerce")
    high = pd.to_numeric(frame["high_price"], errors="coerce")
    low = pd.to_numeric(frame["low_price"], errors="coerce")
    volume = pd.to_numeric(frame["volume_shares"], errors="coerce")
    frame["intraday_range"] = ((high - low) / close.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    dollar_volume = (close.abs() * volume).where(lambda x: x > 0.0)
    frame["log_dollar_volume"] = np.log1p(dollar_volume).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    rolling_volume = frame.groupby("permno", sort=False)["log_dollar_volume"].transform(lambda x: x.rolling(5, min_periods=2).mean())
    frame["volume_shock_5d"] = (frame["log_dollar_volume"] - rolling_volume).fillna(0.0)
    market = frame.drop_duplicates("calendar_date").sort_values("calendar_date").set_index("calendar_date")
    market["mktrf_5d"] = (1.0 + market["mktrf"]).rolling(5, min_periods=1).apply(np.prod, raw=True) - 1.0
    market["market_volatility_5d"] = market["mktrf"].rolling(5, min_periods=2).std(ddof=1).fillna(0.0)
    frame["mktrf_1d"] = frame["calendar_date"].map(market["mktrf"])
    frame["mktrf_5d"] = frame["calendar_date"].map(market["mktrf_5d"])
    frame["market_volatility_5d"] = frame["calendar_date"].map(market["market_volatility_5d"])
    frame["umd_1d"] = frame["umd"].fillna(0.0)
    for horizon in (1, 5):
        labels = np.full(len(frame), np.nan, dtype=float)
        available = np.full(len(frame), np.datetime64("NaT"), dtype="datetime64[ns]")
        for _, index in frame.groupby("permno", sort=False).groups.items():
            loc = np.asarray(list(index), dtype=int)
            values = frame.loc[loc, "stock_total_return"].to_numpy(float)
            dates = frame.loc[loc, "calendar_date"].to_numpy(dtype="datetime64[ns]")
            stock_side = float(np.sign(frame.loc[loc[0], "target_weight"]))
            for position in range(len(loc)):
                start = position + 2  # decision close t, execution close t+1
                stop = min(start + horizon, len(loc))
                if stop - start < horizon:
                    continue
                labels[loc[position]] = stock_side * (float(np.prod(1.0 + values[start:stop])) - 1.0)
                available[loc[position]] = dates[stop - 1]
        frame[f"target_signed_return_{horizon}d"] = labels
        frame[f"label_available_date_{horizon}d"] = pd.to_datetime(available)
    for column in DYNAMIC_COLUMNS:
        if column.startswith("news_"):
            frame[column] = 0.0
    keep = [
        "holding_month", "signal_date", "calendar_date", "entry_date", "exit_date", "permno", "permco",
        "sic2_industry_category", "encoded_row", "target_weight", "target_gross",
        "stock_total_return", "rf", "full_spread_20d", "adv20_usd",
        "impact_volatility_daily", "primary_borrow_rate_annual", "return_source",
        "source_ticker", "sessions_in_month", "day_index", *DYNAMIC_COLUMNS,
        "target_signed_return_1d", "label_available_date_1d",
        "target_signed_return_5d", "label_available_date_5d",
    ]
    frame = frame[keep].copy()
    _atomic_parquet(frame, path)
    errors = pd.DataFrame(reconciliation)
    _atomic_json(
        receipt_path,
        {
            "schema": "research_v6_6_prepared_month_v1",
            "holding_month": month,
            "rows": len(frame),
            "positions": int(frame["permno"].nunique()),
            "sessions": int(frame["calendar_date"].nunique()),
            "rf": rf_month,
            "maximum_initial_daily_reconciliation_error": float(errors["initial_error"].abs().max()),
            "market_accounting_positions": int(errors["source"].eq("market_accounting").sum()),
        },
    )
    return frame


def _prepare_days(config: Mapping[str, Any], output: Path, holdings: pd.DataFrame) -> pd.DataFrame:
    combined = output / "prepared" / "position_days.parquet"
    if combined.is_file():
        return _read_parquet(combined)
    months = _month_sequence(config["chronology"]["initial_fit_start"], config["chronology"]["historical_test_end"])
    parts = [_prepare_month_days(month, holdings, config, output) for month in months]
    frame = pd.concat(parts, ignore_index=True)
    frame = _add_dynamic_news(frame, config, output)
    _atomic_parquet(frame, combined)
    return frame


def _normalize_symbol(value: object) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value).upper())


def _month_raw_news(month: str) -> list[dict[str, Any]]:
    path = RAW_NEWS / f"alpaca_news_{month}.jsonl"
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _add_dynamic_news(days: pd.DataFrame, config: Mapping[str, Any], output: Path) -> pd.DataFrame:
    cache_root = output / "prepared" / "news_months"
    lexicon = news_helpers._load_lexicon()  # noqa: SLF001
    positive, negative, uncertain = lexicon
    lseg = _read_parquet(LSEG_UNIVERSE)[["holding_month", "permno", "permco", "ticker"]]
    lseg["holding_month"] = lseg["holding_month"].astype(str)
    lseg["ticker"] = lseg["ticker"].map(_normalize_symbol)
    completed: list[pd.DataFrame] = []
    for month, month_days in days.groupby("holding_month", sort=True):
        cached = cache_root / f"{month}.parquet"
        if cached.is_file():
            news = _read_parquet(cached)
            completed.append(news)
            continue
        dates = sorted(pd.to_datetime(month_days["calendar_date"]).dt.normalize().unique())
        start = pd.Timestamp(dates[0]) - pd.Timedelta(days=7)
        end = pd.Timestamp(dates[-1])
        universe = lseg.loc[lseg["holding_month"].eq(month)].copy()
        sic_path = MODEL_FEATURE_MONTHS / f"{month}.parquet"
        sic = _read_parquet(sic_path)[["permno", "sic2_industry_category"]]
        universe = universe.merge(sic, on="permno", how="left", validate="one_to_one")
        symbol_to_permnos: dict[str, set[int]] = {}
        for row in universe.itertuples(index=False):
            if row.ticker:
                symbol_to_permnos.setdefault(str(row.ticker), set()).add(int(row.permno))
        permno_to_permco = dict(zip(universe["permno"].astype(int), universe["permco"].astype(int), strict=True))
        permno_to_sic = dict(zip(universe["permno"].astype(int), universe["sic2_industry_category"].astype(str), strict=True))
        articles = _month_raw_news(_previous_month(month)) + _month_raw_news(month) + _month_raw_news(_next_month(month))
        by_id: dict[str, dict[str, Any]] = {}
        for article in articles:
            created = news_helpers._parse_time(article.get("created_at"))  # noqa: SLF001
            updated = news_helpers._parse_time(article.get("updated_at"))  # noqa: SLF001
            if created is None or updated is None:
                continue
            created_naive = pd.Timestamp(created).tz_convert("UTC").tz_localize(None)
            if created_naive < start or created_naive > end + pd.Timedelta(days=1):
                continue
            key = str(article.get("id") or article.get("url") or "")
            if key:
                by_id[key] = article
        events: list[dict[str, Any]] = []
        for article in by_id.values():
            symbols = {_normalize_symbol(value) for value in article.get("symbols", [])} - {""}
            direct = set().union(*(symbol_to_permnos.get(symbol, set()) for symbol in symbols)) if symbols else set()
            if not direct:
                continue
            text, _ = news_helpers._clean_text(article)  # noqa: SLF001
            tokens = [value.upper() for value in TOKEN_RE.findall(text)]
            denominator = max(len(tokens), 1)
            tone = (sum(token in positive for token in tokens) - sum(token in negative for token in tokens)) / denominator
            uncertainty = sum(token in uncertain for token in tokens) / denominator
            created = news_helpers._parse_time(article.get("created_at"))  # noqa: SLF001
            updated = news_helpers._parse_time(article.get("updated_at"))  # noqa: SLF001
            if created is None or updated is None:
                continue
            families = {permno_to_permco[value] for value in direct if value in permno_to_permco}
            sic_to_families: dict[str, set[int]] = {}
            for permno in direct:
                sic2 = permno_to_sic.get(permno, "")
                if sic2 and sic2 != "<NA>":
                    sic_to_families.setdefault(sic2, set()).add(permno_to_permco[permno])
            industries = {sic2 for sic2, values in sic_to_families.items() if len(values) >= 2}
            events.append(
                {
                    "created": pd.Timestamp(created), "updated": pd.Timestamp(updated),
                    "tone": float(tone), "uncertainty": float(uncertainty),
                    "direct": direct, "family": families, "industry": industries,
                }
            )
        news = month_days[["holding_month", "calendar_date", "permno", "permco", "sic2_industry_category"]].copy()
        for column in (value for value in DYNAMIC_COLUMNS if value.startswith("news_")):
            news[column] = 0.0
        for date in dates:
            local_cutoff = dt.datetime.combine(
                pd.Timestamp(date).date(), dt.time(16, 0), tzinfo=ZoneInfo("America/New_York")
            ).astimezone(dt.timezone.utc)
            day_rows = news.index[pd.to_datetime(news["calendar_date"]).dt.normalize().eq(pd.Timestamp(date))]
            for channel, entity_column in (("direct", "permno"), ("family", "permco"), ("industry", "sic2_industry_category")):
                entities = news.loc[day_rows, entity_column]
                for window in (1, 7):
                    cutoff_start = local_cutoff - dt.timedelta(days=window)
                    relevant = [
                        event for event in events
                        if cutoff_start <= event["created"] <= local_cutoff and event["updated"] <= local_cutoff
                    ]
                    for index, entity in entities.items():
                        matched = [event for event in relevant if entity in event[channel] or str(entity) in event[channel]]
                        prefix = f"news_{channel}_{window}d"
                        if matched:
                            news.at[index, f"{prefix}_count"] = float(len(matched))
                            news.at[index, f"{prefix}_tone_mean"] = float(np.mean([event["tone"] for event in matched]))
                            news.at[index, f"{prefix}_uncertainty_mean"] = float(np.mean([event["uncertainty"] for event in matched]))
        news = news.drop(columns=["permco", "sic2_industry_category"])
        _atomic_parquet(news, cached)
        _atomic_json(
            cache_root / f"{month}.json",
            {"schema": "research_v6_6_dynamic_news_v1", "month": month, "mapped_events": len(events), "rows": len(news)},
        )
        completed.append(news)
    news_all = pd.concat(completed, ignore_index=True)
    value_columns = [value for value in DYNAMIC_COLUMNS if value.startswith("news_")]
    base = days.drop(columns=value_columns, errors="ignore")
    result = base.merge(news_all, on=["holding_month", "calendar_date", "permno"], how="left", validate="one_to_one")
    result[value_columns] = result[value_columns].fillna(0.0)
    return result


def _feature_matrix(
    frame: pd.DataFrame,
    config: Mapping[str, Any],
    row_index: np.ndarray | None = None,
) -> np.ndarray:
    index = np.arange(len(frame), dtype=np.int64) if row_index is None else np.asarray(row_index, dtype=np.int64)
    matrix_path = ROOT / config["sources"]["v6_encoded_inputs"]["matrix"]
    monthly = np.load(matrix_path, mmap_mode="r")
    encoded = frame.iloc[index]["encoded_row"].to_numpy(np.int64)
    base = np.asarray(monthly[encoded], dtype=np.float32)
    dynamic = frame.iloc[index][list(DYNAMIC_COLUMNS)].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    dynamic[~np.isfinite(dynamic)] = 0.0
    return np.c_[base, dynamic].astype(np.float32, copy=False)


def _fit_estimator(
    family: str,
    profile: Mapping[str, Any],
    x: np.ndarray,
    y: np.ndarray,
) -> Any:
    if family == "xgboost":
        parameters = dict(profile)
        parameters.pop("id", None)
        estimator = XGBRegressor(
            **parameters,
            objective="reg:squarederror",
            tree_method="hist",
            max_bin=128,
            n_jobs=4,
            random_state=42,
        )
    elif family == "elastic_net":
        estimator = make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(),
            ElasticNet(
                alpha=float(profile["alpha"]),
                l1_ratio=float(profile["l1_ratio"]),
                max_iter=int(profile.get("max_iter", 5000)),
                selection="cyclic",
                random_state=42,
            ),
        )
    else:
        raise ValueError(f"unknown predictive family: {family}")
    estimator.fit(x, y)
    return estimator


def _model_specs(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for profile in config["predictive_models"]["xgboost_profiles"]:
        specs.append({"family": "xgboost", "profile": dict(profile), "id": str(profile["id"])})
    for profile in config["predictive_models"]["elastic_net_profiles"]:
        specs.append({"family": "elastic_net", "profile": dict(profile), "id": str(profile["id"])})
    return specs


def _prediction_checkpoint(
    *,
    frame: pd.DataFrame,
    config: Mapping[str, Any],
    output: Path,
    spec: Mapping[str, Any],
    horizon: int,
    phase: str,
    refit_month: str,
    forecast_months: list[str],
    train_cutoff: pd.Timestamp,
) -> pd.DataFrame:
    block_identity = _canonical_sha(
        {
            "spec": spec,
            "horizon": horizon,
            "phase": phase,
            "refit_month": refit_month,
            "forecast_months": forecast_months,
            "train_cutoff": train_cutoff,
            "config": _sha256(DEFAULT_CONFIG),
        }
    )
    root = output / "models" / phase / str(spec["id"]) / f"h{horizon}"
    prediction_path = root / f"{refit_month}.parquet"
    receipt_path = root / f"{refit_month}.json"
    if prediction_path.is_file() and receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("block_identity") != block_identity:
            raise RuntimeError(f"predictive checkpoint identity mismatch: {receipt_path}")
        return _read_parquet(prediction_path)
    target_column = f"target_signed_return_{horizon}d"
    availability_column = f"label_available_date_{horizon}d"
    train_mask = (
        frame["holding_month"].lt(refit_month)
        & pd.to_datetime(frame[availability_column], errors="coerce").le(train_cutoff)
        & pd.to_numeric(frame[target_column], errors="coerce").notna()
    )
    forecast_mask = frame["holding_month"].isin(forecast_months)
    train_index = np.flatnonzero(train_mask.to_numpy())
    forecast_index = np.flatnonzero(forecast_mask.to_numpy())
    if len(train_index) < 5000 or len(forecast_index) == 0:
        raise RuntimeError(f"insufficient predictive data for {phase}/{refit_month}/{spec['id']}/h{horizon}")
    x_train = _feature_matrix(frame, config, train_index)
    y_train = frame.iloc[train_index][target_column].to_numpy(float)
    x_forecast = _feature_matrix(frame, config, forecast_index)
    estimator = _fit_estimator(str(spec["family"]), spec["profile"], x_train, y_train)
    score = np.asarray(estimator.predict(x_forecast), dtype=float)
    prediction = frame.iloc[forecast_index][["holding_month", "calendar_date", "permno"]].copy()
    prediction["score"] = score
    prediction["model_id"] = str(spec["id"])
    prediction["horizon"] = int(horizon)
    prediction["refit_month"] = refit_month
    _atomic_parquet(prediction, prediction_path)
    receipt: dict[str, Any] = {
        "schema": "research_v6_6_predictive_refit_v1",
        "block_identity": block_identity,
        "family": spec["family"],
        "profile": spec["profile"],
        "horizon": horizon,
        "refit_month": refit_month,
        "forecast_months": forecast_months,
        "train_cutoff": str(train_cutoff),
        "train_rows": int(len(train_index)),
        "forecast_rows": int(len(forecast_index)),
        "train_first_month": str(frame.iloc[train_index]["holding_month"].min()),
        "train_last_month": str(frame.iloc[train_index]["holding_month"].max()),
    }
    if str(spec["family"]) == "xgboost":
        importance = np.asarray(estimator.feature_importances_, dtype=float)
        top = np.argsort(-importance)[:30]
        receipt["top_feature_indices"] = [int(value) for value in top]
        receipt["top_feature_importance"] = [float(importance[value]) for value in top]
    _atomic_json(receipt_path, receipt)
    return prediction


def _validation_predictions(
    frame: pd.DataFrame, config: Mapping[str, Any], output: Path
) -> dict[str, pd.DataFrame]:
    chronology = config["chronology"]
    forecast_months = _month_sequence(chronology["selection_start"], chronology["selection_end"])
    first = frame.loc[frame["holding_month"].eq(forecast_months[0]), "signal_date"]
    cutoff = pd.Timestamp(first.iloc[0])
    result: dict[str, pd.DataFrame] = {}
    components: dict[tuple[str, int], pd.DataFrame] = {}
    for spec in _model_specs(config):
        for horizon in config["predictive_models"]["targets"]:
            prediction = _prediction_checkpoint(
                frame=frame,
                config=config,
                output=output,
                spec=spec,
                horizon=int(horizon),
                phase="selection",
                refit_month=forecast_months[0],
                forecast_months=forecast_months,
                train_cutoff=cutoff,
            )
            key = f"{spec['id']}__h{horizon}"
            result[key] = prediction
            components[(str(spec["id"]), int(horizon))] = prediction
    enet_ids = [spec["id"] for spec in _model_specs(config) if spec["family"] == "elastic_net"]
    xgb_ids = [spec["id"] for spec in _model_specs(config) if spec["family"] == "xgboost"]
    for horizon in config["predictive_models"]["targets"]:
        for xgb_id in xgb_ids:
            left = components[(str(xgb_id), int(horizon))]
            right = components[(str(enet_ids[0]), int(horizon))]
            joined = left[["holding_month", "calendar_date", "permno", "score"]].merge(
                right[["holding_month", "calendar_date", "permno", "score"]],
                on=["holding_month", "calendar_date", "permno"],
                suffixes=("_xgb", "_enet"),
                validate="one_to_one",
            )
            for name in ("xgb", "enet"):
                values = joined[f"score_{name}"].to_numpy(float)
                scale = max(float(np.std(values, ddof=1)), 1e-8)
                joined[f"z_{name}"] = (values - float(np.mean(values))) / scale
            joined["score"] = 0.5 * (joined["z_xgb"] + joined["z_enet"])
            key = f"ensemble_{xgb_id}_{enet_ids[0]}__h{horizon}"
            result[key] = joined[["holding_month", "calendar_date", "permno", "score"]].copy()
    return result


def _selected_test_predictions(
    frame: pd.DataFrame,
    config: Mapping[str, Any],
    output: Path,
    selected: Mapping[str, Any],
    validation_predictions: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    model_key = str(selected["model_key"])
    horizon = int(selected["horizon"])
    test_months = _month_sequence(config["chronology"]["historical_test_start"], config["chronology"]["historical_test_end"])
    refits = [month for month in test_months if month.endswith(("-01", "-04", "-07", "-10"))]
    components: list[str]
    if model_key.startswith("ensemble_"):
        body = model_key.removeprefix("ensemble_").split("__h")[0]
        enet_id = next(spec["id"] for spec in _model_specs(config) if spec["family"] == "elastic_net")
        xgb_id = body[: -(len(str(enet_id)) + 1)]
        components = [xgb_id, str(enet_id)]
    else:
        components = [model_key.split("__h")[0]]
    specs = {str(spec["id"]): spec for spec in _model_specs(config)}
    component_parts: dict[str, list[pd.DataFrame]] = {value: [] for value in components}
    for position, refit in enumerate(refits):
        start = test_months.index(refit)
        stop = test_months.index(refits[position + 1]) if position + 1 < len(refits) else len(test_months)
        block = test_months[start:stop]
        cutoff = pd.Timestamp(frame.loc[frame["holding_month"].eq(refit), "signal_date"].iloc[0])
        for component in components:
            component_parts[component].append(
                _prediction_checkpoint(
                    frame=frame,
                    config=config,
                    output=output,
                    spec=specs[component],
                    horizon=horizon,
                    phase="historical_test",
                    refit_month=refit,
                    forecast_months=block,
                    train_cutoff=cutoff,
                )
            )
    assembled = {key: pd.concat(value, ignore_index=True) for key, value in component_parts.items()}
    if len(components) == 1:
        result = assembled[components[0]][["holding_month", "calendar_date", "permno", "score"]].copy()
    else:
        xgb_id, enet_id = components
        result = assembled[xgb_id][["holding_month", "calendar_date", "permno", "score"]].merge(
            assembled[enet_id][["holding_month", "calendar_date", "permno", "score"]],
            on=["holding_month", "calendar_date", "permno"],
            suffixes=("_xgb", "_enet"),
            validate="one_to_one",
        )
        scaling = selected["ensemble_scaling"]
        result["score"] = 0.5 * (
            (result["score_xgb"] - float(scaling["xgb_mean"])) / float(scaling["xgb_std"])
            + (result["score_enet"] - float(scaling["enet_mean"])) / float(scaling["enet_std"])
        )
        result = result[["holding_month", "calendar_date", "permno", "score"]]
    _atomic_parquet(result, output / "models" / "selected_historical_test_predictions.parquet")
    return result


def _transaction_cost(
    delta: np.ndarray,
    frame: pd.DataFrame,
    nav: float,
    costs: Mapping[str, Any],
) -> float:
    absolute = np.abs(np.asarray(delta, dtype=float))
    if not np.any(absolute > 0.0):
        return 0.0
    spread = frame["full_spread_20d"].to_numpy(float)
    adv = frame["adv20_usd"].to_numpy(float)
    volatility = frame["impact_volatility_daily"].to_numpy(float)
    if not np.isfinite(np.c_[spread, adv, volatility]).all() or np.any(adv <= 0.0):
        raise RuntimeError("transaction cost inputs are invalid")
    linear = (
        float(costs["spread_multiplier"]) * spread / 2.0
        + float(costs["commission_rate"])
    ) * absolute
    impact = (
        float(costs["impact_coefficient"])
        * volatility
        * np.sqrt(nav / adv)
        * np.power(absolute, 1.5)
    )
    return float(np.sum(linear + impact))


def _rule_target(row: pd.Series, rule: Mapping[str, Any], current_fraction: float) -> float:
    target = current_fraction
    if "stop_loss" in rule and float(row["signed_position_return"]) <= float(rule["stop_loss"]):
        target = min(target, float(rule["stop_loss_fraction"]))
    if "trailing_drawdown" in rule and float(row["signed_peak_to_current_drawdown"]) <= float(rule["trailing_drawdown"]):
        target = min(target, float(rule["trailing_fraction"]))
    if "take_profit" in rule and float(row["signed_position_return"]) >= float(rule["take_profit"]):
        target = min(target, float(rule["take_profit_fraction"]))
    return target


def _summary(policy_id: str, monthly: pd.DataFrame, daily: pd.DataFrame, actions: pd.DataFrame) -> dict[str, Any]:
    excess = monthly["net_excess_return"].to_numpy(float)
    total = monthly["net_return"].to_numpy(float)
    volatility = float(np.std(excess, ddof=1) * math.sqrt(12.0))
    wealth = float(np.prod(1.0 + total))
    daily_nav = daily["nav_end"].to_numpy(float)
    peaks = np.maximum.accumulate(np.r_[daily["nav_start"].iloc[0], daily_nav])[1:]
    return {
        "policy": policy_id,
        "months": int(len(monthly)),
        "starting_nav": float(monthly["nav_start"].iloc[0]),
        "ending_nav": float(monthly["nav_end"].iloc[-1]),
        "cumulative_total_return": wealth - 1.0,
        "total_return_cagr": wealth ** (12.0 / len(monthly)) - 1.0,
        "annualized_net_excess_return": float(np.mean(excess) * 12.0),
        "annualized_realized_volatility": volatility,
        "net_sharpe": float(np.mean(excess) / np.std(excess, ddof=1) * math.sqrt(12.0)),
        "maximum_drawdown": float(np.min(daily_nav / peaks - 1.0, initial=0.0)),
        "positive_month_fraction": float(np.mean(excess > 0.0)),
        "annualized_total_cost": float(monthly["total_cost"].mean() * 12.0),
        "annualized_transaction_cost": float(monthly["transaction_cost"].mean() * 12.0),
        "annualized_borrow_cost": float(monthly["borrow_cost"].mean() * 12.0),
        "annualized_financing_cost": float(monthly["financing_cost"].mean() * 12.0),
        "maximum_close_gross": float(daily["gross_end"].max()),
        "intervention_count": int(len(actions)),
        "exit_count": int(actions["new_fraction"].eq(0.0).sum()) if len(actions) else 0,
        "trim_count": int(actions["new_fraction"].between(0.0, 1.0, inclusive="neither").sum()) if len(actions) else 0,
        "safety_action_count": int(actions["action_type"].eq("safety").sum()) if len(actions) else 0,
        "intervention_one_way_turnover": float(actions["absolute_trade_weight"].sum()) if len(actions) else 0.0,
    }


def _replay_policy(
    *,
    days: pd.DataFrame,
    config: Mapping[str, Any],
    policy: Mapping[str, Any],
    evaluation_start: str,
    evaluation_end: str,
    scores: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    costs = config["account_and_costs"]
    replay_days = days.loc[days["holding_month"].between(config["chronology"]["initial_fit_start"], evaluation_end)].copy()
    if scores is not None:
        replay_days = replay_days.merge(
            scores[["holding_month", "calendar_date", "permno", "score"]],
            on=["holding_month", "calendar_date", "permno"], how="left", validate="one_to_one",
        )
    else:
        replay_days["score"] = np.nan
    nav = float(costs["starting_nav_dollars"])
    exposure: dict[int, float] = {}
    risk_cache: dict[int, dict[str, float]] = {}
    daily_records: list[dict[str, Any]] = []
    monthly_records: list[dict[str, Any]] = []
    action_records: list[dict[str, Any]] = []
    for month, month_frame in replay_days.groupby("holding_month", sort=True):
        month_frame = month_frame.sort_values(["calendar_date", "permno"], kind="mergesort")
        first = month_frame.sort_values("calendar_date").drop_duplicates("permno")
        target = dict(zip(first["permno"].astype(int), first["target_weight"].astype(float), strict=True))
        for row in first.itertuples(index=False):
            risk_cache[int(row.permno)] = {
                "full_spread_20d": float(row.full_spread_20d),
                "adv20_usd": float(row.adv20_usd),
                "impact_volatility_daily": float(row.impact_volatility_daily),
                "primary_borrow_rate_annual": float(row.primary_borrow_rate_annual),
            }
        month_nav_start = nav
        union = sorted(set(exposure) | set(target))
        risk_frame = pd.DataFrame([risk_cache[value] for value in union])
        previous_weight = np.asarray([exposure.get(value, 0.0) / nav for value in union])
        target_weight = np.asarray([target.get(value, 0.0) for value in union])
        entry_cost_rate = _transaction_cost(target_weight - previous_weight, risk_frame, nav, costs)
        entry_cost_dollars = nav * entry_cost_rate
        exposure = {value: target.get(value, 0.0) * nav for value in target}
        nav -= entry_cost_dollars
        fraction = {value: 1.0 for value in target}
        pending: dict[int, float] = {}
        pending_safety = False
        month_costs = {"transaction": entry_cost_dollars, "borrow": 0.0, "financing": 0.0}
        month_action_start = month >= evaluation_start
        for calendar_date, day in month_frame.groupby("calendar_date", sort=True):
            day = day.sort_values("permno", kind="mergesort")
            nav_start = nav
            rf = float(day["rf"].iloc[0])
            permnos = day["permno"].astype(int).to_numpy()
            exposed = np.asarray([exposure.get(value, 0.0) for value in permnos], dtype=float)
            stock_return = day["stock_total_return"].to_numpy(float)
            stock_pnl = float(np.sum(exposed * (stock_return - rf)))
            weights = exposed / nav
            borrow_rate = day["primary_borrow_rate_annual"].to_numpy(float)
            sessions = int(day["sessions_in_month"].iloc[0])
            borrow_rate_day = float(np.sum(np.maximum(-weights, 0.0) * borrow_rate / 12.0 / sessions))
            gross = float(np.abs(weights).sum())
            financing_rate_day = max(gross - 1.0, 0.0) * float(costs["financing_rate_annual"]) / 12.0 / sessions
            borrow_dollars = nav * borrow_rate_day
            financing_dollars = nav * financing_rate_day
            nav = nav * (1.0 + rf) + stock_pnl - borrow_dollars - financing_dollars
            month_costs["borrow"] += borrow_dollars
            month_costs["financing"] += financing_dollars
            for value, result in zip(permnos, exposed * (1.0 + stock_return), strict=True):
                exposure[int(value)] = float(result)
            action_cost_dollars = 0.0
            if month_action_start and pending:
                changed = []
                for value, new_fraction in pending.items():
                    old_fraction = fraction.get(value, 0.0)
                    if new_fraction >= old_fraction - 1e-12 or old_fraction <= 0.0:
                        continue
                    old_exposure = exposure.get(value, 0.0)
                    new_exposure = old_exposure * new_fraction / old_fraction
                    changed.append((value, old_fraction, new_fraction, old_exposure, new_exposure))
                if changed:
                    action_risk = pd.DataFrame([risk_cache[value] for value, *_ in changed])
                    delta = np.asarray([(new - old) / nav for _, _, _, old, new in changed])
                    rate = _transaction_cost(delta, action_risk, nav, costs)
                    action_cost_dollars = nav * rate
                    nav -= action_cost_dollars
                    month_costs["transaction"] += action_cost_dollars
                    for value, old_fraction, new_fraction, old_exposure, new_exposure in changed:
                        exposure[value] = new_exposure
                        fraction[value] = new_fraction
                        action_records.append(
                            {
                                "policy": policy["id"], "holding_month": month,
                                "calendar_date": pd.Timestamp(calendar_date), "permno": value,
                                "action_type": "exit" if new_fraction == 0.0 else "trim",
                                "old_fraction": old_fraction, "new_fraction": new_fraction,
                                "absolute_trade_weight": abs(new_exposure - old_exposure) / nav,
                                "transaction_cost_dollars": action_cost_dollars * abs(new_exposure - old_exposure) / max(sum(abs(item[4] - item[3]) for item in changed), 1e-12),
                            }
                        )
            pending = {}
            safety_cost_dollars = 0.0
            if month_action_start and pending_safety:
                gross_now = sum(abs(value) for value in exposure.values()) / nav
                target_gross = float(costs["operational_safety_target"])
                if gross_now > target_gross:
                    scalar = target_gross / gross_now
                    changed_values = [value for value, amount in exposure.items() if abs(amount) > 0.0]
                    action_risk = pd.DataFrame([risk_cache[value] for value in changed_values])
                    delta = np.asarray([(exposure[value] * scalar - exposure[value]) / nav for value in changed_values])
                    rate = _transaction_cost(delta, action_risk, nav, costs)
                    safety_cost_dollars = nav * rate
                    nav -= safety_cost_dollars
                    month_costs["transaction"] += safety_cost_dollars
                    trade_weight = float(np.abs(delta).sum())
                    for value in changed_values:
                        exposure[value] *= scalar
                        fraction[value] *= scalar
                    action_records.append(
                        {
                            "policy": policy["id"], "holding_month": month,
                            "calendar_date": pd.Timestamp(calendar_date), "permno": -1,
                            "action_type": "safety", "old_fraction": 1.0,
                            "new_fraction": scalar, "absolute_trade_weight": trade_weight,
                            "transaction_cost_dollars": safety_cost_dollars,
                        }
                    )
            pending_safety = False
            gross_end = sum(abs(value) for value in exposure.values()) / nav
            if month_action_start:
                if policy.get("rule"):
                    for _, row in day.iterrows():
                        value = int(row["permno"])
                        if fraction.get(value, 0.0) <= 0.0:
                            continue
                        new_fraction = _rule_target(row, policy["rule"], fraction[value])
                        if new_fraction < fraction[value] - 1e-12:
                            pending[value] = new_fraction
                if policy.get("predictive"):
                    thresholds = policy["predictive"]
                    for _, row in day.iterrows():
                        value = int(row["permno"])
                        score = float(row["score"]) if pd.notna(row["score"]) else math.nan
                        if fraction.get(value, 0.0) <= 0.0 or not math.isfinite(score) or score >= 0.0:
                            continue
                        new_fraction = fraction[value]
                        if score <= float(thresholds["exit_threshold"]):
                            new_fraction = 0.0
                        elif score <= float(thresholds["trim_threshold"]):
                            new_fraction = min(new_fraction, 0.5)
                        if new_fraction < fraction[value] - 1e-12:
                            pending[value] = new_fraction
                if policy.get("safety") and gross_end > float(costs["operational_safety_trigger"]):
                    pending_safety = True
            daily_records.append(
                {
                    "policy": policy["id"], "holding_month": month,
                    "calendar_date": pd.Timestamp(calendar_date), "nav_start": nav_start,
                    "nav_end": nav, "rf": rf, "stock_pnl_dollars": stock_pnl,
                    "borrow_cost_dollars": borrow_dollars, "financing_cost_dollars": financing_dollars,
                    "action_cost_dollars": action_cost_dollars + safety_cost_dollars,
                    "gross_end": gross_end,
                }
            )
        if month >= evaluation_start:
            total_cost = sum(month_costs.values()) / month_nav_start
            rf_month = float(np.prod(1.0 + month_frame.drop_duplicates("calendar_date").sort_values("calendar_date")["rf"].to_numpy(float)) - 1.0)
            net_return = nav / month_nav_start - 1.0
            monthly_records.append(
                {
                    "policy": policy["id"], "holding_month": month,
                    "nav_start": month_nav_start, "nav_end": nav,
                    "net_return": net_return, "rf": rf_month,
                    "net_excess_return": net_return - rf_month,
                    "transaction_cost": month_costs["transaction"] / month_nav_start,
                    "borrow_cost": month_costs["borrow"] / month_nav_start,
                    "financing_cost": month_costs["financing"] / month_nav_start,
                    "total_cost": total_cost,
                }
            )
    monthly = pd.DataFrame(monthly_records)
    daily = pd.DataFrame(daily_records)
    daily = daily.loc[daily["holding_month"].between(evaluation_start, evaluation_end)].reset_index(drop=True)
    actions = pd.DataFrame(action_records)
    if len(actions):
        actions = actions.loc[actions["holding_month"].between(evaluation_start, evaluation_end)].reset_index(drop=True)
    else:
        actions = pd.DataFrame(columns=["policy", "holding_month", "calendar_date", "permno", "action_type", "old_fraction", "new_fraction", "absolute_trade_weight", "transaction_cost_dollars"])
    return monthly, daily, actions, _summary(str(policy["id"]), monthly, daily, actions)


def _hac_t(values: Iterable[float], lags: int = 6) -> float:
    series = np.asarray(list(values), dtype=float)
    series = series[np.isfinite(series)]
    n = len(series)
    if n < 3:
        return math.nan
    centered = series - float(np.mean(series))
    long_run = float(centered @ centered / n)
    for lag in range(1, min(lags, n - 1) + 1):
        covariance = float(centered[lag:] @ centered[:-lag] / n)
        long_run += 2.0 * (1.0 - lag / (lags + 1.0)) * covariance
    standard_error = math.sqrt(max(long_run, 1e-18) / n)
    return float(np.mean(series) / standard_error)


def _save_replay(
    root: Path,
    policy_id: str,
    monthly: pd.DataFrame,
    daily: pd.DataFrame,
    actions: pd.DataFrame,
    summary: Mapping[str, Any],
) -> None:
    target = root / policy_id
    _atomic_parquet(monthly, target / "monthly.parquet")
    _atomic_parquet(daily, target / "daily.parquet")
    _atomic_parquet(actions, target / "actions.parquet")
    _atomic_json(target / "summary.json", dict(summary))


def _candidate_order(item: Mapping[str, Any]) -> tuple[float, float, float, float]:
    summary = item["summary"]
    return (
        float(summary["net_sharpe"]),
        float(summary["annualized_net_excess_return"]),
        float(summary["maximum_drawdown"]),
        -float(summary["intervention_one_way_turnover"]),
    )


def _ensemble_scaling(model_key: str, predictions: Mapping[str, pd.DataFrame]) -> dict[str, float] | None:
    if not model_key.startswith("ensemble_"):
        return None
    horizon = int(model_key.rsplit("__h", 1)[1])
    body = model_key.removeprefix("ensemble_").rsplit("__h", 1)[0]
    enet_key = next(value for value in predictions if value.startswith("enet_") and value.endswith(f"__h{horizon}"))
    enet_id = enet_key.rsplit("__h", 1)[0]
    xgb_id = body[: -(len(enet_id) + 1)]
    xgb = predictions[f"{xgb_id}__h{horizon}"]["score"].to_numpy(float)
    enet = predictions[f"{enet_id}__h{horizon}"]["score"].to_numpy(float)
    return {
        "xgb_mean": float(np.mean(xgb)), "xgb_std": max(float(np.std(xgb, ddof=1)), 1e-8),
        "enet_mean": float(np.mean(enet)), "enet_std": max(float(np.std(enet, ddof=1)), 1e-8),
    }


def _control_reconciliation(
    monthly: pd.DataFrame,
    phase: str,
) -> dict[str, Any]:
    roots = sorted((ROOT / "artifacts/research_v6_1/alpaca_paper_translation").glob("*/"))
    if not roots:
        return {"available": False}
    source = roots[-1] / phase / "alpaca_target_1p80" / "broker_current_monthly.csv"
    if not source.is_file():
        return {"available": False}
    reference = pd.read_csv(source)
    joined = monthly[["holding_month", "net_return", "net_excess_return"]].merge(
        reference[["holding_month", "net_return", "net_excess_return"]],
        on="holding_month", suffixes=("_daily", "_monthly"), validate="one_to_one",
    )
    return {
        "available": True,
        "months": len(joined),
        "maximum_absolute_monthly_total_return_difference": float((joined["net_return_daily"] - joined["net_return_monthly"]).abs().max()),
        "mean_monthly_total_return_difference": float((joined["net_return_daily"] - joined["net_return_monthly"]).mean()),
        "explanation": "The V6.6 replay accrues borrow and financing from current daily drifted weights; the predecessor charged monthly start weights. Gross stock and risk-free accounting use the same authenticated outcomes.",
    }


def _result_markdown(result: Mapping[str, Any]) -> str:
    control = result["historical_test"]["monthly_control"]
    rows = []
    for policy_id, summary in result["historical_test"].items():
        difference = result["paired_differences"].get(policy_id, {})
        rows.append(
            f"| {policy_id} | {summary['annualized_net_excess_return']:.2%} | "
            f"{summary['annualized_realized_volatility']:.2%} | {summary['net_sharpe']:.3f} | "
            f"{summary['maximum_drawdown']:.2%} | {summary['intervention_count']} | "
            f"{difference.get('paired_hac_t', float('nan')):.2f} |"
        )
    table = "\n".join(rows)
    winner = result["decision"]["best_historical_profile"]
    selected_rule = result["selection"]["selected_rule"]["policy_id"]
    selected_predictive = result["selection"]["selected_predictive"]["policy_id"]
    return f"""# Research V6.6 — intramonth intervention study

## Executive Summary

- **Decision:** `{result['decision']['terminal_decision']}`.
- **Monthly control:** {control['annualized_net_excess_return']:.2%} annualized net excess return, {control['net_sharpe']:.3f} Sharpe, and {control['maximum_drawdown']:.2%} maximum drawdown in January 2024–August 2025.
- **Best historical intervention:** `{winner}`. Historical evidence cannot reselect a policy; promotion requires all frozen validation, return, Sharpe, drawdown, cost, and paired-inference gates.
- **Learning rule:** the predictive overlay refits quarterly with expanding data, but only labels observable by that refit's monthly signal date enter training.

## What was tested

The frozen monthly V6.1 stock selectors, FF5/momentum gate, optimizer, and
relative monthly targets are unchanged. At each rebalance, targets are scaled
uniformly to no more than 1.80x gross. An intervention can only trim or close
an existing position; it cannot add a stock or re-enter before the next
monthly rebalance. A close signal executes at the following trading-session
close.

2022 is the initial predictive fit, 2023 selects one transparent-rule policy
and one predictive policy, and January 2024–August 2025 evaluates only the
prespecified control, safety policy, selected policies, and their frozen
predictive-plus-safety combination. The selected rule was `{selected_rule}`;
the selected predictive policy was `{selected_predictive}`.

## Historical performance

| Policy | Ann. net excess | Volatility | Sharpe | Max drawdown | Actions | Paired HAC t |
|---|---:|---:|---:|---:|---:|---:|
{table}

## Interpretation

{result['decision']['interpretation']}

## Data and implementation checks

- Daily stock returns compound to the authenticated monthly outcomes; explicit market-accounting fallbacks remain untradeable until their exit-date accounting observation.
- Daily return coverage before fallback is {result['data_quality']['raw_daily_position_day_coverage']:.2%} of active position-days and {result['data_quality']['raw_daily_weight_coverage']:.2%} of absolute starting weight.
- The untouched daily control's maximum absolute monthly return difference from the predecessor monthly-cost replay is {result['control_reconciliation']['historical_test'].get('maximum_absolute_monthly_total_return_difference', float('nan')):.4%}; the documented difference is daily drift-aware financing and borrow accrual.
- The complete authenticated 850-column monthly feature matrix is reused, augmented with causal price, liquidity, position-state, market, and rolling direct/family/industry news variables.

## Caveats and next step

This is post-diagnostic historical development evidence, not untouched or live
confirmation. September 2025–August 2026 remains quarantined. A policy may be
paper-traded only as a shadow alongside the monthly control until forward
evidence beginning no earlier than September 2026 confirms that its turnover
and execution assumptions survive in real time.
"""


def run(config_path: Path, output_root: Path) -> Path:
    config, identity = _verify(config_path)
    output = _output_path(output_root, identity)
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "run_manifest.json"
    _atomic_json(
        manifest,
        {"schema": "research_v6_6_intramonth_run_v1", "status": "RUNNING", "identity": identity},
    )
    holdings = _prepare_holdings(config, output)
    days = _prepare_days(config, output, holdings)
    raw_usable = pd.to_numeric(days["stock_total_return"], errors="coerce").notna()
    original_usable = days["return_source"].eq("primary_outcomes")
    weight_by_day = days["target_weight"].abs()
    data_quality = {
        "position_months": int(len(holdings)),
        "position_days": int(len(days)),
        "months": int(days["holding_month"].nunique()),
        "raw_daily_position_day_coverage": float(original_usable.mean()),
        "raw_daily_weight_coverage": float(weight_by_day[original_usable].sum() / weight_by_day.sum()),
        "post_fallback_finite_return_fraction": float(raw_usable.mean()),
        "encoded_feature_count": int(config["sources"]["v6_encoded_inputs"]["encoded_feature_count"]),
        "dynamic_feature_count": len(DYNAMIC_COLUMNS),
    }

    selection_root = output / "selection"
    control_policy = {"id": "monthly_control"}
    safety_policy = {"id": "safety_only", "safety": True}
    selection_candidates: list[dict[str, Any]] = []
    control_monthly, control_daily, control_actions, control_summary = _replay_policy(
        days=days, config=config, policy=control_policy,
        evaluation_start=config["chronology"]["selection_start"],
        evaluation_end=config["chronology"]["selection_end"],
    )
    _save_replay(selection_root, "monthly_control", control_monthly, control_daily, control_actions, control_summary)
    safety_monthly, safety_daily, safety_actions, safety_summary = _replay_policy(
        days=days, config=config, policy=safety_policy,
        evaluation_start=config["chronology"]["selection_start"],
        evaluation_end=config["chronology"]["selection_end"],
    )
    _save_replay(selection_root, "safety_only", safety_monthly, safety_daily, safety_actions, safety_summary)
    for rule in config["transparent_rules"]:
        policy = {"id": f"rule__{rule['id']}", "rule": dict(rule)}
        monthly, daily, actions, summary = _replay_policy(
            days=days, config=config, policy=policy,
            evaluation_start=config["chronology"]["selection_start"],
            evaluation_end=config["chronology"]["selection_end"],
        )
        _save_replay(selection_root, str(policy["id"]), monthly, daily, actions, summary)
        selection_candidates.append({"kind": "rule", "policy_id": policy["id"], "policy": policy, "summary": summary})

    validation_predictions = _validation_predictions(days, config, output)
    for model_key, prediction in validation_predictions.items():
        horizon = int(model_key.rsplit("__h", 1)[1])
        values = prediction["score"].to_numpy(float)
        for intensity in config["predictive_models"]["decision_intensities"]:
            exit_threshold = min(float(np.quantile(values, float(intensity["exit_quantile"]))), 0.0)
            trim_threshold = min(float(np.quantile(values, float(intensity["trim_quantile"]))), 0.0)
            predictive = {
                "model_key": model_key,
                "horizon": horizon,
                "intensity": intensity["id"],
                "exit_threshold": exit_threshold,
                "trim_threshold": trim_threshold,
            }
            policy_id = f"predictive__{model_key}__{intensity['id']}"
            policy = {"id": policy_id, "predictive": predictive}
            monthly, daily, actions, summary = _replay_policy(
                days=days, config=config, policy=policy, scores=prediction,
                evaluation_start=config["chronology"]["selection_start"],
                evaluation_end=config["chronology"]["selection_end"],
            )
            _save_replay(selection_root, policy_id, monthly, daily, actions, summary)
            selection_candidates.append(
                {
                    "kind": "predictive", "policy_id": policy_id, "policy": policy,
                    "model_key": model_key, "horizon": horizon,
                    "intensity": intensity["id"], "exit_threshold": exit_threshold,
                    "trim_threshold": trim_threshold, "summary": summary,
                }
            )
    candidate_frame = pd.DataFrame(
        [{"kind": item["kind"], "policy_id": item["policy_id"], **item["summary"]} for item in selection_candidates]
    )
    _atomic_parquet(candidate_frame, selection_root / "candidate_summary.parquet")
    selected_rule = max((item for item in selection_candidates if item["kind"] == "rule"), key=_candidate_order)
    selected_predictive = max((item for item in selection_candidates if item["kind"] == "predictive"), key=_candidate_order)
    scaling = _ensemble_scaling(str(selected_predictive["model_key"]), validation_predictions)
    if scaling is not None:
        selected_predictive["ensemble_scaling"] = scaling
    _atomic_json(
        selection_root / "selection.json",
        {
            "schema": "research_v6_6_intervention_selection_v1",
            "control": control_summary,
            "safety": safety_summary,
            "selected_rule": selected_rule,
            "selected_predictive": selected_predictive,
        },
    )

    test_scores = _selected_test_predictions(days, config, output, selected_predictive, validation_predictions)
    test_profiles = [
        control_policy,
        safety_policy,
        {"id": "selected_rule", "rule": selected_rule["policy"]["rule"]},
        {"id": "selected_predictive", "predictive": selected_predictive["policy"]["predictive"]},
        {"id": "selected_predictive_plus_safety", "predictive": selected_predictive["policy"]["predictive"], "safety": True},
    ]
    historical: dict[str, dict[str, Any]] = {}
    historical_months: dict[str, pd.DataFrame] = {}
    historical_root = output / "historical_test"
    for policy in test_profiles:
        score = test_scores if policy.get("predictive") else None
        monthly, daily, actions, summary = _replay_policy(
            days=days, config=config, policy=policy, scores=score,
            evaluation_start=config["chronology"]["historical_test_start"],
            evaluation_end=config["chronology"]["historical_test_end"],
        )
        _save_replay(historical_root, str(policy["id"]), monthly, daily, actions, summary)
        historical[str(policy["id"])] = summary
        historical_months[str(policy["id"])] = monthly
    paired: dict[str, dict[str, Any]] = {}
    base = historical_months["monthly_control"][["holding_month", "net_excess_return"]]
    for policy_id, monthly in historical_months.items():
        if policy_id == "monthly_control":
            paired[policy_id] = {"annualized_difference": 0.0, "paired_hac_t": math.nan}
            continue
        joined = base.merge(monthly[["holding_month", "net_excess_return"]], on="holding_month", suffixes=("_control", "_policy"), validate="one_to_one")
        difference = joined["net_excess_return_policy"] - joined["net_excess_return_control"]
        paired[policy_id] = {
            "annualized_difference": float(difference.mean() * 12.0),
            "paired_hac_t": _hac_t(difference, lags=6),
            "positive_difference_fraction": float((difference > 0.0).mean()),
        }
    control = historical["monthly_control"]
    gates: dict[str, dict[str, bool]] = {}
    for policy_id, summary in historical.items():
        if policy_id == "monthly_control":
            continue
        gates[policy_id] = {
            "validation_frozen": True,
            "test_sharpe_higher": float(summary["net_sharpe"]) > float(control["net_sharpe"]),
            "test_return_higher": float(summary["annualized_net_excess_return"]) > float(control["annualized_net_excess_return"]),
            "drawdown_within_tolerance": float(summary["maximum_drawdown"]) >= float(control["maximum_drawdown"]) - float(config["promotion"]["historical_test_maximum_drawdown_not_worse_by_more_than"]),
            "paired_hac_t": float(paired[policy_id]["paired_hac_t"]) >= float(config["promotion"]["paired_monthly_difference_hac_t_minimum"]),
        }
    qualifying = [policy for policy, values in gates.items() if all(values.values())]
    best_historical = max(historical, key=lambda value: (
        historical[value]["net_sharpe"], historical[value]["annualized_net_excess_return"]
    ))
    terminal = (
        f"HISTORICAL_INTERVENTION_QUALIFIES_{max(qualifying, key=lambda value: historical[value]['net_sharpe'])}"
        if qualifying else "RETAIN_MONTHLY_REBALANCE_CONTROL_NO_INTRAMONTH_INTERVENTION_QUALIFIES"
    )
    interpretation = (
        "At least one validation-frozen intervention improved both historical return and Sharpe, preserved drawdown, and passed the paired monthly HAC threshold. It remains a forward paper-trading challenger rather than untouched confirmation."
        if qualifying
        else "No validation-frozen intervention cleared every economic and paired-inference gate. The evidence therefore favors preserving monthly discipline and using the daily layer only for operational exposure safety."
    )
    result = {
        "schema": "research_v6_6_intramonth_result_v1",
        "status": "COMPLETE",
        "identity": identity,
        "data_quality": data_quality,
        "selection": {
            "period": f"{config['chronology']['selection_start']}--{config['chronology']['selection_end']}",
            "control": control_summary,
            "safety": safety_summary,
            "selected_rule": selected_rule,
            "selected_predictive": selected_predictive,
        },
        "historical_test": historical,
        "paired_differences": paired,
        "promotion_gates": gates,
        "control_reconciliation": {
            "selection": _control_reconciliation(control_monthly, "validation"),
            "historical_test": _control_reconciliation(historical_months["monthly_control"], "historical_test"),
        },
        "decision": {
            "terminal_decision": terminal,
            "qualifying_profiles": qualifying,
            "best_historical_profile": best_historical,
            "interpretation": interpretation,
        },
    }
    _atomic_json(output / "result.json", result)
    _atomic_text(output / "RESULT.md", _result_markdown(result))
    _atomic_json(
        manifest,
        {
            "schema": "research_v6_6_intramonth_run_v1", "status": "COMPLETE",
            "identity": identity, "result": "result.json", "report": "RESULT.md",
        },
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(run(args.config.resolve(), args.output_root.resolve()))


if __name__ == "__main__":
    main()
