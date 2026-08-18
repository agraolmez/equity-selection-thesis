"""Point-in-time Alpaca related-news features for prospective V6.8 signals."""

from __future__ import annotations

import datetime as dt
import hashlib
import html
import json
import os
import re
from pathlib import Path
from typing import Any, Iterable, Mapping
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from portfolio_bot.research_v6.news import AlpacaNewsClient, AlpacaNewsCredentials
from portfolio_bot.research_v6_8.credentials import load_paper_credentials


WINDOWS = (1, 7, 30)
CHANNELS = ("direct", "family", "industry")
TAG_RE = re.compile(r"<[^>]+>")
TOKEN_RE = re.compile(r"[A-Za-z]+")
SPACE_RE = re.compile(r"\s+")


class LiveNewsFeatureError(RuntimeError):
    """The live point-in-time news block failed closed."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _normalize_symbol(value: object) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(value).upper())


def _parse_time(value: object) -> dt.datetime | None:
    try:
        return dt.datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(
            dt.timezone.utc
        )
    except (TypeError, ValueError):
        return None


def _clean_text(article: Mapping[str, Any]) -> tuple[str, bool]:
    segments: list[str] = []
    fingerprints: set[str] = set()
    for name in ("headline", "summary", "content"):
        raw = str(article.get(name) or "")
        normalized = SPACE_RE.sub(" ", html.unescape(TAG_RE.sub(" ", raw))).strip()
        fingerprint = normalized.casefold()
        if normalized and fingerprint not in fingerprints:
            segments.append(normalized)
            fingerprints.add(fingerprint)
    return " ".join(segments), bool(str(article.get("content") or "").strip())


def _load_lexicon(path: Path) -> tuple[set[str], set[str], set[str]]:
    if not path.is_file():
        raise LiveNewsFeatureError(f"news lexicon is missing: {path}")
    dictionary = pd.read_csv(
        path, usecols=["Word", "Negative", "Positive", "Uncertainty"]
    )
    words = dictionary["Word"].astype(str).str.upper()
    return tuple(
        set(words.loc[pd.to_numeric(dictionary[column], errors="coerce").gt(0)])
        for column in ("Positive", "Negative", "Uncertainty")
    )  # type: ignore[return-value]


def _empty() -> dict[str, float]:
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
        values = aggregates.setdefault((channel, entity, window), _empty())
        values["count"] += 1.0
        values["tone_sum"] += tone
        values["uncertainty_sum"] += uncertainty
        values["recency_days"] = min(values["recency_days"], age_days)
        values["full_content_count"] += float(full_content)
        values["duplicate_count"] += float(duplicate)


def fetch_point_in_time_articles(
    client: AlpacaNewsClient,
    *,
    start: dt.datetime,
    cutoff: dt.datetime,
) -> tuple[list[dict[str, Any]], int]:
    token: str | None = None
    pages = 0
    by_id: dict[str, dict[str, Any]] = {}
    while True:
        payload = client.fetch_page(
            start=start.isoformat().replace("+00:00", "Z"),
            end=cutoff.isoformat().replace("+00:00", "Z"),
            page_token=token,
        )
        pages += 1
        for value in payload.get("news", []):
            article = dict(value)
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
            if key:
                by_id[key] = article
        token = payload.get("next_page_token") or None
        if token is None:
            break
        token = str(token)
    return sorted(
        by_id.values(), key=lambda row: (str(row.get("created_at")), str(row.get("id")))
    ), pages


def build_live_news_features(
    *,
    universe: pd.DataFrame,
    identity: pd.DataFrame,
    articles: Iterable[Mapping[str, Any]],
    cutoff: dt.datetime,
    lexicon: tuple[set[str], set[str], set[str]],
) -> tuple[pd.DataFrame, dict[str, int]]:
    article_rows = [dict(value) for value in articles]
    required_universe = {"live_security_id", "TickerSymbol", "issuer_key"}
    if not required_universe.issubset(universe.columns):
        raise LiveNewsFeatureError("live news universe omits identity columns")
    if "sic2_industry_category" not in identity:
        raise LiveNewsFeatureError("live news identity block omits SIC2")
    stocks = universe[["live_security_id", "TickerSymbol", "issuer_key"]].merge(
        identity[["live_security_id", "sic2_industry_category"]],
        on="live_security_id",
        how="left",
        validate="one_to_one",
    )
    stocks["symbol_key"] = stocks["TickerSymbol"].map(_normalize_symbol)
    symbol_to_ids = stocks.groupby("symbol_key")["live_security_id"].agg(set).to_dict()
    indexed = stocks.set_index("live_security_id")
    positive, negative, uncertain = lexicon
    aggregates: dict[tuple[str, object, int], dict[str, float]] = {}
    text_fingerprints: set[str] = set()
    mapped_articles = 0
    full_content_articles = 0
    for article in article_rows:
        created = _parse_time(article.get("created_at"))
        updated = _parse_time(article.get("updated_at"))
        if created is None or updated is None or created > cutoff or updated > cutoff:
            continue
        symbols = {
            _normalize_symbol(value) for value in (article.get("symbols") or [])
        } - {""}
        direct_ids: set[str] = set()
        for symbol in symbols:
            direct_ids.update(map(str, symbol_to_ids.get(symbol, set())))
        if not direct_ids:
            continue
        text, full_content = _clean_text(article)
        tokens = [token.upper() for token in TOKEN_RE.findall(text)]
        denominator = max(len(tokens), 1)
        tone = (
            sum(token in positive for token in tokens)
            - sum(token in negative for token in tokens)
        ) / denominator
        uncertainty = sum(token in uncertain for token in tokens) / denominator
        fingerprint = hashlib.sha256(
            SPACE_RE.sub(" ", text.lower()).encode("utf-8")
        ).hexdigest()
        duplicate = fingerprint in text_fingerprints
        text_fingerprints.add(fingerprint)
        age = max((cutoff - created).total_seconds() / 86_400.0, 0.0)
        direct = indexed.loc[sorted(direct_ids)]
        issuer_keys = set(direct["issuer_key"].dropna().astype(str))
        industry_issuers: dict[str, set[str]] = {}
        for row in direct.itertuples():
            sic2 = (
                ""
                if pd.isna(row.sic2_industry_category)
                else str(row.sic2_industry_category)
            )
            if sic2 and sic2 not in {"<NA>", "UNKNOWN_INDUSTRY"}:
                industry_issuers.setdefault(sic2, set()).add(str(row.issuer_key))
        industries = {
            sic2 for sic2, issuers in industry_issuers.items() if len(issuers) >= 2
        }
        for security_id in direct_ids:
            _add(aggregates, channel="direct", entity=security_id, age_days=age,
                 tone=tone, uncertainty=uncertainty, full_content=full_content,
                 duplicate=duplicate)
        for issuer in issuer_keys:
            _add(aggregates, channel="family", entity=issuer, age_days=age,
                 tone=tone, uncertainty=uncertainty, full_content=full_content,
                 duplicate=duplicate)
        for sic2 in industries:
            _add(aggregates, channel="industry", entity=sic2, age_days=age,
                 tone=tone, uncertainty=uncertainty, full_content=full_content,
                 duplicate=duplicate)
        mapped_articles += 1
        full_content_articles += int(full_content)

    result = stocks[["live_security_id"]].copy()
    result["news_source_available"] = np.int8(1)
    for channel in CHANNELS:
        entities = (
            stocks["live_security_id"]
            if channel == "direct"
            else stocks["issuer_key"]
            if channel == "family"
            else stocks["sic2_industry_category"]
        )
        for window in WINDOWS:
            rows = [aggregates.get((channel, value, window)) for value in entities]
            count = np.asarray([row["count"] if row else 0.0 for row in rows])
            prefix = f"news_{channel}_{window}d"
            result[f"{prefix}_count"] = count.astype(np.float32)
            result[f"{prefix}_tone_mean"] = np.asarray(
                [row["tone_sum"] / row["count"] if row else 0.0 for row in rows],
                dtype=np.float32,
            )
            result[f"{prefix}_uncertainty_mean"] = np.asarray(
                [row["uncertainty_sum"] / row["count"] if row else 0.0 for row in rows],
                dtype=np.float32,
            )
            result[f"{prefix}_recency_days"] = np.asarray(
                [row["recency_days"] if row else 0.0 for row in rows], dtype=np.float32
            )
            result[f"{prefix}_duplicate_share"] = np.asarray(
                [row["duplicate_count"] / row["count"] if row else 0.0 for row in rows],
                dtype=np.float32,
            )
            result[f"{prefix}_full_content_share"] = np.asarray(
                [row["full_content_count"] / row["count"] if row else 0.0 for row in rows],
                dtype=np.float32,
            )
    count_columns = [column for column in result if column.endswith("_count")]
    result["news_related"] = result[count_columns].gt(0).any(axis=1).astype("int8")
    return result, {
        "articles": len(article_rows),
        "mapped_articles": mapped_articles,
        "full_content_mapped_articles": full_content_articles,
        "related_rows": int(result["news_related"].sum()),
    }


def materialize_live_news_features(
    *,
    config_path: Path,
    universe_path: Path,
    identity_path: Path,
    registry_path: Path,
    output: Path,
    now: dt.datetime | None = None,
) -> dict[str, Any]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("schema") != "research_v6_8_live_news_features_v1":
        raise LiveNewsFeatureError("live news config schema differs")
    universe = pd.read_parquet(universe_path)
    identity = pd.read_parquet(identity_path)
    registry = pd.read_parquet(registry_path)
    signals = pd.to_datetime(universe["signal_date"], errors="raise").dt.date
    if signals.nunique() != 1:
        raise LiveNewsFeatureError("live news universe must contain one signal date")
    cutoff_time = dt.time.fromisoformat(str(config["signal_cutoff_new_york"]))
    cutoff = dt.datetime.combine(
        signals.iloc[0], cutoff_time, tzinfo=ZoneInfo("America/New_York")
    ).astimezone(dt.timezone.utc)
    observed_now = (now or dt.datetime.now(dt.timezone.utc)).astimezone(dt.timezone.utc)
    if observed_now < cutoff:
        raise LiveNewsFeatureError("signal-close news cutoff has not occurred yet")
    start = cutoff - dt.timedelta(days=int(config["lookback_days"]))
    credentials = load_paper_credentials(
        key_environment=str(config["credentials"]["key_environment"]),
        secret_environment=str(config["credentials"]["secret_environment"]),
    )
    client = AlpacaNewsClient(
        AlpacaNewsCredentials(
            key_id=credentials.key_id,
            secret_key=credentials.secret_key,
        )
    )
    articles, pages = fetch_point_in_time_articles(client, start=start, cutoff=cutoff)
    output.mkdir(parents=True, exist_ok=True)
    raw_path = output / "articles.jsonl"
    temporary = raw_path.with_suffix(".jsonl.tmp")
    with temporary.open("w", encoding="utf-8") as destination:
        for article in articles:
            destination.write(json.dumps(article, sort_keys=True) + "\n")
    os.replace(temporary, raw_path)
    features, stats = build_live_news_features(
        universe=universe,
        identity=identity,
        articles=articles,
        cutoff=cutoff,
        lexicon=_load_lexicon(Path(config["lexicon_path"])),
    )
    expected = registry.loc[
        registry["model_ordinal"].between(248, 303), "feature_name"
    ].astype(str).tolist()
    if len(expected) != 56 or set(features.columns) != {"live_security_id", *expected}:
        raise LiveNewsFeatureError("live news feature schema differs from the frozen block")
    data_path = output / "news_features.parquet"
    _atomic_parquet(data_path, features[["live_security_id", *expected]])
    receipt = {
        "schema": "research_v6_8_live_news_features_receipt_v1",
        "status": "PASS",
        "signal_date": signals.iloc[0].isoformat(),
        "cutoff_utc": cutoff.isoformat(),
        "rows": int(len(features)),
        "feature_count": 56,
        "pages": int(pages),
        **stats,
        "raw_sha256": _sha256(raw_path),
        "output_sha256": _sha256(data_path),
        "credential_source": credentials.source,
        "credentials_in_artifact": False,
    }
    _atomic_json(output / "receipt.json", receipt)
    return receipt


__all__ = [
    "LiveNewsFeatureError",
    "build_live_news_features",
    "fetch_point_in_time_articles",
    "materialize_live_news_features",
]
