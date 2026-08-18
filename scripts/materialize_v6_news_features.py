#!/usr/bin/env python3
"""Materialize the V6 point-in-time related-news block month by month.

The raw Alpaca archive is immutable.  This builder uses only articles whose
creation and latest available update are no later than the 16:00 New York
signal-close cutoff.  It maps Alpaca symbols to the historical CRSP security,
issuer (PERMCO), and SIC2 observed at that signal.  Industry propagation is
allowed only when an article tags at least two distinct issuers in the same
SIC2.  No-news rows are neutral zeros; downstream V6 models use the block only
as a residual adjustment on rows with related news.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import html
import json
import os
from pathlib import Path
import re
from typing import Iterable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


DATABASE = Path("/Users/agraemirolmez/Desktop/Thesis")
RAW_NEWS = DATABASE / "raw/research_v6/alpaca_news"
LM_DICTIONARY = DATABASE / (
    "raw/research_v6/public_sources/lm_dictionary/"
    "Loughran-McDonald_MasterDictionary_1993-2025.csv"
)
OUTPUT = DATABASE / "processed/research_v6/news_features_v2"
LSEG_UNIVERSE = DATABASE / "raw/research_v6/lseg/input_universe.parquet"
VALIDATION_FEATURES = DATABASE / (
    "interim/phase_05_validation/stage11-validation-data-v1/"
    "candidate_payloads/validation_feature_store_raw.parquet"
)
FINAL_FEATURES = DATABASE / (
    "processed/phase_06_final_test/stage13-final-inputs-v2/primary_predictors.parquet"
)
DECEMBER_CONTEXT = DATABASE / (
    "cache/phase_06_final_test/stage13-final-source-replay-v1/"
    "dbccbf9cd220d890d8c025bb6e92854bf61098f1eb42517da934c62b98e7adf5/"
    "formula_work/feature_sources/universe/internal_context_signal_date_universe.parquet"
)
IDENTITY_INTERVAL = DATABASE / (
    "cache/phase_06_final_test/stage13-final-source-replay-v1/"
    "dbccbf9cd220d890d8c025bb6e92854bf61098f1eb42517da934c62b98e7adf5/"
    "formula_work/feature_sources/universe_context/typed/"
    "crsp_identity_interval.parquet"
)

WINDOWS = (1, 7, 30)
CHANNELS = ("direct", "family", "industry")
TAG_RE = re.compile(r"<[^>]+>")
TOKEN_RE = re.compile(r"[A-Za-z]+")
SPACE_RE = re.compile(r"\s+")


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


def _holding_month(signal: pd.Series) -> pd.Series:
    return (pd.to_datetime(signal).dt.to_period("M") + 1).astype(str)


def _normalize_symbol(value: object) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value).upper())


def _load_universe() -> pd.DataFrame:
    lseg = pd.read_parquet(LSEG_UNIVERSE).loc[
        :, ["holding_month", "signal_date", "permno", "permco", "ticker", "issuer_name"]
    ]
    lseg["holding_month"] = lseg["holding_month"].astype(str)
    lseg["signal_date"] = pd.to_datetime(lseg["signal_date"])
    validation = pd.read_parquet(
        VALIDATION_FEATURES,
        columns=["signal_date", "permno", "sic2_industry_category"],
    )
    validation.insert(0, "holding_month", _holding_month(validation["signal_date"]))
    final = pd.read_parquet(
        FINAL_FEATURES,
        columns=["holding_month", "signal_date", "permno", "sic2_industry_category"],
    )
    final["holding_month"] = final["holding_month"].astype(str)
    features = pd.concat([validation, final], ignore_index=True)
    features = features.rename(columns={"sic2_industry_category": "sic2_category"})
    features["signal_date"] = pd.to_datetime(features["signal_date"])
    features = features.drop_duplicates(["holding_month", "permno"])
    universe = lseg.merge(
        features[["holding_month", "permno", "sic2_category"]],
        on=["holding_month", "permno"],
        how="left",
        validate="one_to_one",
    )

    identity = pd.read_parquet(
        IDENTITY_INTERVAL,
        columns=[
            "permno",
            "start_date",
            "end_date",
            "ticker",
            "issuer_name",
            "sic_code",
        ],
    )
    signal = pd.Timestamp("2019-11-29")
    identity_start = pd.to_datetime(identity["start_date"])
    identity_end = pd.to_datetime(identity["end_date"])
    valid_identity = identity.loc[
        (identity_start.isna() | identity_start.le(signal))
        & (identity_end.isna() | identity_end.ge(signal))
    ].copy()
    valid_identity = valid_identity.sort_values(
        ["permno", "start_date"], kind="mergesort"
    ).drop_duplicates("permno", keep="last")
    december_identity = valid_identity[["permno", "ticker", "issuer_name", "sic_code"]].copy()
    december_identity["sic2_identity"] = pd.to_numeric(
        december_identity["sic_code"], errors="coerce"
    ).floordiv(100).astype("Int64").astype("string")
    december = universe["holding_month"].eq("2019-12")
    if not december.any():
        raise RuntimeError("LSEG V6 universe omits December 2019")
    december_fill = universe.loc[december, ["permno"]].merge(
        december_identity,
        on="permno",
        how="left",
        validate="one_to_one",
    )
    universe.loc[december, "sic2_category"] = december_fill["sic2_identity"].to_numpy()
    universe.loc[december & universe["ticker"].isna(), "ticker"] = december_fill.loc[
        universe.loc[december, "ticker"].isna().to_numpy(), "ticker"
    ].to_numpy()
    universe.loc[december & universe["issuer_name"].isna(), "issuer_name"] = december_fill.loc[
        universe.loc[december, "issuer_name"].isna().to_numpy(), "issuer_name"
    ].to_numpy()
    universe["ticker"] = universe["ticker"].map(_normalize_symbol)
    universe["sic2_category"] = universe["sic2_category"].astype("string")
    universe["permno"] = pd.to_numeric(universe["permno"]).astype("int64")
    universe["permco"] = pd.to_numeric(universe["permco"]).astype("int64")
    universe = universe.loc[
        universe["holding_month"].between("2015-01", "2025-08")
        & universe["ticker"].ne("")
    ].copy()
    if universe.duplicated(["holding_month", "permno"]).any():
        raise RuntimeError("V6 news universe contains duplicate stock-months")
    return universe.sort_values(["holding_month", "permno"], kind="mergesort")


def _load_lexicon() -> tuple[set[str], set[str], set[str]]:
    dictionary = pd.read_csv(
        LM_DICTIONARY,
        usecols=["Word", "Negative", "Positive", "Uncertainty"],
    )
    words = dictionary["Word"].astype(str).str.upper()
    return tuple(
        set(words.loc[pd.to_numeric(dictionary[column], errors="coerce").gt(0)])
        for column in ("Positive", "Negative", "Uncertainty")
    )  # type: ignore[return-value]


def _calendar_months(start: dt.date, end: dt.date) -> Iterable[str]:
    current = dt.date(start.year, start.month, 1)
    terminal = dt.date(end.year, end.month, 1)
    while current <= terminal:
        yield f"{current.year:04d}-{current.month:02d}"
        current = (
            dt.date(current.year + 1, 1, 1)
            if current.month == 12
            else dt.date(current.year, current.month + 1, 1)
        )


def _parse_time(value: object) -> dt.datetime | None:
    try:
        return dt.datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(
            dt.timezone.utc
        )
    except (TypeError, ValueError):
        return None


def _clean_text(article: dict[str, object]) -> tuple[str, bool]:
    content = str(article.get("content") or "")
    full = bool(content.strip())
    # Alpaca/Benzinga content can repeat the headline or summary verbatim.
    # Count each normalized segment once so duplicated payload structure does
    # not mechanically amplify tone or uncertainty.
    segments: list[str] = []
    fingerprints: set[str] = set()
    for name in ("headline", "summary", "content"):
        segment = str(article.get(name) or "")
        normalized = SPACE_RE.sub(" ", html.unescape(TAG_RE.sub(" ", segment))).strip()
        fingerprint = normalized.casefold()
        if normalized and fingerprint not in fingerprints:
            segments.append(normalized)
            fingerprints.add(fingerprint)
    raw = " ".join(segments)
    cleaned = html.unescape(TAG_RE.sub(" ", raw))
    return SPACE_RE.sub(" ", cleaned).strip(), full


def _article_rows(
    *, start: dt.datetime, cutoff: dt.datetime
) -> tuple[list[dict[str, object]], int]:
    by_id: dict[str, dict[str, object]] = {}
    raw_rows = 0
    for month in _calendar_months(start.date(), cutoff.date()):
        path = RAW_NEWS / f"alpaca_news_{month}.jsonl"
        if not path.is_file():
            if month == "2014-12":
                continue
            raise FileNotFoundError(f"missing Alpaca news month: {month}")
        with path.open("r", encoding="utf-8") as source:
            for line in source:
                raw_rows += 1
                article = json.loads(line)
                created = _parse_time(article.get("created_at"))
                updated = _parse_time(article.get("updated_at"))
                if (
                    created is None
                    or updated is None
                    or created < start
                    or created > cutoff
                    or updated > cutoff
                ):
                    continue
                key = str(article.get("id") or article.get("url") or "")
                if not key:
                    continue
                existing = by_id.get(key)
                if existing is None or str(article.get("updated_at")) < str(existing.get("updated_at")):
                    by_id[key] = article
    ordered = sorted(
        by_id.values(), key=lambda value: (str(value.get("created_at")), str(value.get("id")))
    )
    return ordered, raw_rows


def _empty_accumulator() -> dict[str, float]:
    return {
        "count": 0.0,
        "tone_sum": 0.0,
        "uncertainty_sum": 0.0,
        "recency_days": 1e9,
        "full_content_count": 0.0,
        "duplicate_count": 0.0,
    }


def _add(
    aggregates: dict[tuple[str, object, int], dict[str, float]],
    *,
    channel: str,
    entity: object,
    age_days: float,
    tone: float,
    uncertainty: float,
    full_content: bool,
    duplicate: bool,
) -> None:
    for window in WINDOWS:
        if age_days > window:
            continue
        values = aggregates.setdefault((channel, entity, window), _empty_accumulator())
        values["count"] += 1.0
        values["tone_sum"] += tone
        values["uncertainty_sum"] += uncertainty
        values["recency_days"] = min(values["recency_days"], age_days)
        values["full_content_count"] += float(full_content)
        values["duplicate_count"] += float(duplicate)


def _materialize_month(
    month: str,
    stocks: pd.DataFrame,
    lexicon: tuple[set[str], set[str], set[str]],
) -> tuple[pd.DataFrame, dict[str, object]]:
    signal_date = pd.Timestamp(stocks["signal_date"].iloc[0]).date()
    cutoff = dt.datetime.combine(
        signal_date, dt.time(16, 0), tzinfo=ZoneInfo("America/New_York")
    ).astimezone(dt.timezone.utc)
    start = cutoff - dt.timedelta(days=30)
    articles, raw_rows = _article_rows(start=start, cutoff=cutoff)
    symbol_to_permnos: dict[str, set[int]] = {}
    permno_to_permco = dict(zip(stocks["permno"], stocks["permco"], strict=True))
    permno_to_sic2 = dict(zip(stocks["permno"], stocks["sic2_category"], strict=True))
    for row in stocks.itertuples(index=False):
        symbol_to_permnos.setdefault(str(row.ticker), set()).add(int(row.permno))

    positive, negative, uncertain = lexicon
    aggregates: dict[tuple[str, object, int], dict[str, float]] = {}
    fingerprints: set[str] = set()
    eligible_articles = 0
    mapped_articles = 0
    full_content_articles = 0
    for article in articles:
        symbols = {
            _normalize_symbol(value) for value in (article.get("symbols") or [])
        } - {""}
        direct_permnos = set().union(
            *(symbol_to_permnos.get(symbol, set()) for symbol in symbols)
        ) if symbols else set()
        if not direct_permnos:
            continue
        eligible_articles += 1
        text, full_content = _clean_text(article)
        tokens = [value.upper() for value in TOKEN_RE.findall(text)]
        denominator = max(len(tokens), 1)
        positive_count = sum(value in positive for value in tokens)
        negative_count = sum(value in negative for value in tokens)
        uncertainty_count = sum(value in uncertain for value in tokens)
        tone = (positive_count - negative_count) / denominator
        uncertainty = uncertainty_count / denominator
        fingerprint = hashlib.sha256(
            SPACE_RE.sub(" ", text.lower()).encode("utf-8")
        ).hexdigest()
        duplicate = fingerprint in fingerprints
        fingerprints.add(fingerprint)
        created = _parse_time(article.get("created_at"))
        assert created is not None
        age_days = max((cutoff - created).total_seconds() / 86_400.0, 0.0)
        direct_permcos = {permno_to_permco[value] for value in direct_permnos}
        sic_to_permcos: dict[str, set[int]] = {}
        for permno in direct_permnos:
            sic2 = str(permno_to_sic2.get(permno) or "")
            if sic2 and sic2 != "<NA>":
                sic_to_permcos.setdefault(sic2, set()).add(permno_to_permco[permno])
        industry_sic2 = {
            sic2 for sic2, permcos in sic_to_permcos.items() if len(permcos) >= 2
        }
        for permno in direct_permnos:
            _add(
                aggregates,
                channel="direct",
                entity=permno,
                age_days=age_days,
                tone=tone,
                uncertainty=uncertainty,
                full_content=full_content,
                duplicate=duplicate,
            )
        for permco in direct_permcos:
            _add(
                aggregates,
                channel="family",
                entity=permco,
                age_days=age_days,
                tone=tone,
                uncertainty=uncertainty,
                full_content=full_content,
                duplicate=duplicate,
            )
        for sic2 in industry_sic2:
            _add(
                aggregates,
                channel="industry",
                entity=sic2,
                age_days=age_days,
                tone=tone,
                uncertainty=uncertainty,
                full_content=full_content,
                duplicate=duplicate,
            )
        mapped_articles += 1
        full_content_articles += int(full_content)

    result = stocks[["holding_month", "signal_date", "permno", "permco"]].copy()
    result["news_source_available"] = np.int8(signal_date >= dt.date(2015, 1, 1))
    for channel in CHANNELS:
        entities = (
            result["permno"]
            if channel == "direct"
            else result["permco"]
            if channel == "family"
            else stocks["sic2_category"]
        )
        for window in WINDOWS:
            values = [aggregates.get((channel, entity, window)) for entity in entities]
            count = np.asarray([value["count"] if value else 0.0 for value in values])
            prefix = f"news_{channel}_{window}d"
            result[f"{prefix}_count"] = count.astype(np.float32)
            result[f"{prefix}_tone_mean"] = np.asarray(
                [value["tone_sum"] / value["count"] if value else 0.0 for value in values],
                dtype=np.float32,
            )
            result[f"{prefix}_uncertainty_mean"] = np.asarray(
                [value["uncertainty_sum"] / value["count"] if value else 0.0 for value in values],
                dtype=np.float32,
            )
            result[f"{prefix}_recency_days"] = np.asarray(
                [value["recency_days"] if value else 0.0 for value in values],
                dtype=np.float32,
            )
            result[f"{prefix}_duplicate_share"] = np.asarray(
                [value["duplicate_count"] / value["count"] if value else 0.0 for value in values],
                dtype=np.float32,
            )
            result[f"{prefix}_full_content_share"] = np.asarray(
                [value["full_content_count"] / value["count"] if value else 0.0 for value in values],
                dtype=np.float32,
            )
    count_columns = [value for value in result.columns if value.endswith("_count")]
    result["news_related"] = result[count_columns].gt(0).any(axis=1).astype("int8")
    manifest = {
        "schema": "research_v6_related_news_month_v2",
        "holding_month": month,
        "signal_date": signal_date.isoformat(),
        "signal_cutoff_utc": cutoff.isoformat(),
        "universe_rows": len(result),
        "raw_rows_scanned": raw_rows,
        "point_in_time_articles": len(articles),
        "mapped_articles": mapped_articles,
        "full_content_mapped_articles": full_content_articles,
        "related_stock_rows": int(result["news_related"].sum()),
        "industry_rule": "at least two distinct tagged PERMCOs in the same historical SIC2",
        "no_news_policy": "neutral_zero_and_residual_adjustment_only",
        "source_available": bool(signal_date >= dt.date(2015, 1, 1)),
        "text_policy": "unique_normalized_headline_summary_content_segments",
    }
    return result, manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    month_root = output / "months"
    manifest_root = output / "manifests"
    month_root.mkdir(parents=True, exist_ok=True)
    manifest_root.mkdir(parents=True, exist_ok=True)
    universe = _load_universe()
    lexicon = _load_lexicon()
    months = sorted(universe["holding_month"].unique())
    for ordinal, month in enumerate(months, start=1):
        destination = month_root / f"{month}.parquet"
        manifest_path = manifest_root / f"{month}.json"
        if destination.is_file() and manifest_path.is_file():
            receipt = json.loads(manifest_path.read_text())
            if receipt.get("sha256") == _sha256(destination):
                print(f"{ordinal}/{len(months)} {month}: reused", flush=True)
                continue
        stocks = universe.loc[universe["holding_month"].eq(month)].copy()
        frame, receipt = _materialize_month(month, stocks, lexicon)
        _atomic_parquet(frame, destination)
        receipt["sha256"] = _sha256(destination)
        receipt["feature_columns"] = len(frame.columns) - 4
        _atomic_json(manifest_path, receipt)
        print(
            f"{ordinal}/{len(months)} {month}: {receipt['mapped_articles']:,} mapped articles; "
            f"{receipt['related_stock_rows']:,}/{receipt['universe_rows']:,} related rows",
            flush=True,
        )
    manifests = [json.loads(path.read_text()) for path in sorted(manifest_root.glob("*.json"))]
    if len(manifests) != len(months):
        raise RuntimeError("news month manifest coverage is incomplete")
    _atomic_json(
        output / "manifest.json",
        {
            "schema": "research_v6_related_news_block_v2",
            "status": "COMPLETE",
            "first_holding_month": months[0],
            "last_holding_month": months[-1],
            "holding_months": len(months),
            "rows": sum(int(value["universe_rows"]) for value in manifests),
            "related_rows": sum(int(value["related_stock_rows"]) for value in manifests),
            "mapped_articles_month_sum": sum(int(value["mapped_articles"]) for value in manifests),
            "lm_dictionary_sha256": _sha256(LM_DICTIONARY),
            "raw_month_manifest_sha256s": {
                path.name[12:19]: _sha256(path)
                for path in sorted(RAW_NEWS.glob("alpaca_news_*.manifest.json"))
            },
            "month_outputs": {value["holding_month"]: value["sha256"] for value in manifests},
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
