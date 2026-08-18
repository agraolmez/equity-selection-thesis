"""Checkpointed, read-only acquisition of Alpaca historical news.

The client is deliberately restricted to Alpaca's market-data news endpoint.
Credentials are read at runtime and are never included in artifacts, receipts,
exceptions, or request URLs.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


NEWS_ENDPOINT = "https://data.alpaca.markets/v1beta1/news"
KEY_ENV = "APCA_API_KEY_ID"
SECRET_ENV = "APCA_API_SECRET_KEY"


class AlpacaNewsError(RuntimeError):
    """An authenticated Alpaca news request or checkpoint operation failed."""


@dataclasses.dataclass(frozen=True, slots=True)
class AlpacaNewsCredentials:
    key_id: str = dataclasses.field(repr=False)
    secret_key: str = dataclasses.field(repr=False)

    def __post_init__(self) -> None:
        if not self.key_id.strip() or not self.secret_key.strip():
            raise ValueError("Alpaca key ID and secret key must both be non-empty")


@dataclasses.dataclass(frozen=True, slots=True)
class NewsDownloadResult:
    month: str
    output_path: Path
    manifest_path: Path
    article_count: int
    page_count: int
    sha256: str
    resumed: bool


def _parse_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"invalid credentials line in {path.name}")
        name, value = line.split("=", 1)
        values[name.strip()] = value.strip().strip('"').strip("'")
    return values


def load_alpaca_credentials(
    credentials_file: Path | str | None = None,
) -> AlpacaNewsCredentials:
    """Load credentials from a private file overlaying process environment."""

    values = {KEY_ENV: os.environ.get(KEY_ENV, ""), SECRET_ENV: os.environ.get(SECRET_ENV, "")}
    if credentials_file is not None:
        source = Path(credentials_file).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Alpaca credentials file not found: {source}")
        values.update(_parse_env_file(source))
    missing = [name for name in (KEY_ENV, SECRET_ENV) if not values.get(name, "").strip()]
    if missing:
        raise AlpacaNewsError(
            "missing Alpaca credentials; configure " + " and ".join(missing)
        )
    return AlpacaNewsCredentials(
        key_id=values[KEY_ENV].strip(), secret_key=values[SECRET_ENV].strip()
    )


class AlpacaNewsClient:
    """Small standard-library client restricted to one read-only endpoint."""

    def __init__(
        self,
        credentials: AlpacaNewsCredentials,
        *,
        timeout_seconds: float = 30.0,
        maximum_retries: int = 5,
    ) -> None:
        self._credentials = credentials
        self.timeout_seconds = float(timeout_seconds)
        self.maximum_retries = int(maximum_retries)

    def fetch_page(
        self,
        *,
        start: str,
        end: str,
        page_token: str | None = None,
    ) -> Mapping[str, Any]:
        query: dict[str, object] = {
            "start": start,
            "end": end,
            "sort": "asc",
            "limit": 50,
            "include_content": "true",
            "exclude_contentless": "false",
        }
        if page_token:
            query["page_token"] = page_token
        url = NEWS_ENDPOINT + "?" + urlencode(query)
        request = Request(
            url,
            method="GET",
            headers={
                "Accept": "application/json",
                "APCA-API-KEY-ID": self._credentials.key_id,
                "APCA-API-SECRET-KEY": self._credentials.secret_key,
                "User-Agent": "portfolio-bot-thesis-v6-news/1",
            },
        )
        for attempt in range(self.maximum_retries + 1):
            try:
                with urlopen(request, timeout=self.timeout_seconds) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                if not isinstance(payload, dict) or not isinstance(payload.get("news", []), list):
                    raise AlpacaNewsError("Alpaca returned an invalid news payload")
                return payload
            except HTTPError as exc:
                if exc.code not in {429, 500, 502, 503, 504} or attempt >= self.maximum_retries:
                    raise AlpacaNewsError(f"Alpaca news request failed with HTTP {exc.code}") from exc
                retry_after = exc.headers.get("Retry-After")
                delay = float(retry_after) if retry_after else min(2.0**attempt, 30.0)
                time.sleep(max(delay, 0.25))
            except (TimeoutError, URLError) as exc:
                if attempt >= self.maximum_retries:
                    raise AlpacaNewsError("Alpaca news request failed after retries") from exc
                time.sleep(min(2.0**attempt, 30.0))
        raise AssertionError("unreachable")


def _month_bounds(month: str) -> tuple[str, str]:
    try:
        start = dt.date.fromisoformat(month + "-01")
    except ValueError as exc:
        raise ValueError(f"invalid YYYY-MM month: {month!r}") from exc
    if start.month == 12:
        end = dt.date(start.year + 1, 1, 1)
    else:
        end = dt.date(start.year, start.month + 1, 1)
    # Alpaca documents ``end`` as inclusive.  Adjacent monthly snapshots can
    # therefore share an article timestamped exactly at midnight; downstream
    # materialization deduplicates by immutable article ID.  Keeping the clean
    # month boundary also makes an interrupted request token safely resumable.
    return start.isoformat() + "T00:00:00Z", end.isoformat() + "T00:00:00Z"


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def download_news_month(
    client: AlpacaNewsClient,
    *,
    month: str,
    output_root: Path | str,
) -> NewsDownloadResult:
    """Download one calendar month, resuming after its last durable page."""

    start, end = _month_bounds(month)
    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    final_path = root / f"alpaca_news_{month}.jsonl"
    manifest_path = root / f"alpaca_news_{month}.manifest.json"
    partial_path = root / f"alpaca_news_{month}.jsonl.part"
    checkpoint_path = root / f"alpaca_news_{month}.checkpoint.json"

    if final_path.is_file() and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        observed_hash = _sha256_file(final_path)
        if observed_hash != manifest.get("sha256"):
            raise AlpacaNewsError(f"completed news month {month} failed its hash check")
        return NewsDownloadResult(
            month=month,
            output_path=final_path,
            manifest_path=manifest_path,
            article_count=int(manifest["article_count"]),
            page_count=int(manifest["page_count"]),
            sha256=observed_hash,
            resumed=True,
        )

    page_token: str | None = None
    article_count = 0
    page_count = 0
    durable_bytes = 0
    resumed = False
    if checkpoint_path.is_file():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint.get("month") != month or checkpoint.get("start") != start or checkpoint.get("end") != end:
            raise AlpacaNewsError(f"checkpoint identity mismatch for {month}")
        page_token = checkpoint.get("next_page_token") or None
        article_count = int(checkpoint["article_count"])
        page_count = int(checkpoint["page_count"])
        durable_bytes = int(checkpoint["durable_bytes"])
        if not partial_path.is_file():
            raise AlpacaNewsError(f"checkpoint for {month} has no partial data file")
        with partial_path.open("r+b") as output:
            output.truncate(durable_bytes)
        resumed = True
    else:
        partial_path.write_bytes(b"")

    while True:
        payload = client.fetch_page(start=start, end=end, page_token=page_token)
        articles = payload.get("news", [])
        with partial_path.open("ab") as output:
            for article in articles:
                output.write(
                    json.dumps(article, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
                    + b"\n"
                )
            output.flush()
            os.fsync(output.fileno())
            durable_bytes = output.tell()
        article_count += len(articles)
        page_count += 1
        next_token = payload.get("next_page_token") or None
        _atomic_json(
            checkpoint_path,
            {
                "schema": "research_v6_alpaca_news_checkpoint_v1",
                "month": month,
                "start": start,
                "end": end,
                "article_count": article_count,
                "page_count": page_count,
                "durable_bytes": durable_bytes,
                "next_page_token": next_token,
            },
        )
        if next_token is None:
            break
        page_token = str(next_token)

    os.replace(partial_path, final_path)
    digest = _sha256_file(final_path)
    _atomic_json(
        manifest_path,
        {
            "schema": "research_v6_alpaca_news_month_v1",
            "source": NEWS_ENDPOINT,
            "month": month,
            "requested_start": start,
            "requested_end_exclusive": end,
            "include_content": True,
            "article_count": article_count,
            "page_count": page_count,
            "sha256": digest,
            "credentials_persisted": False,
        },
    )
    checkpoint_path.unlink(missing_ok=True)
    return NewsDownloadResult(
        month=month,
        output_path=final_path,
        manifest_path=manifest_path,
        article_count=article_count,
        page_count=page_count,
        sha256=digest,
        resumed=resumed,
    )


__all__ = [
    "AlpacaNewsClient",
    "AlpacaNewsCredentials",
    "AlpacaNewsError",
    "NEWS_ENDPOINT",
    "NewsDownloadResult",
    "download_news_month",
    "load_alpaca_credentials",
]
