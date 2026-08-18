#!/usr/bin/env python3
"""Download public research inputs for the V6 feature investigation.

The downloader is intentionally independent from the existing Alpaca backfill.
Each completed file is published atomically, validated, and recorded in a
source-level manifest. Re-running the command resumes from valid local files.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_ROOT = Path("/Users/agraemirolmez/Desktop/Thesis/raw/research_v6/public_sources")
SEC_INSIDER_PAGE = (
    "https://www.sec.gov/data-research/sec-markets-data/"
    "insider-transactions-data-sets"
)
SEC_FTD_PAGE = "https://www.sec.gov/data-research/sec-markets-data/fails-deliver-data"
TNIC_BASE = "https://hobergphillips.tuck.dartmouth.edu/"
TNIC_FILES = (
    "idata/tnic3_data.zip",
    "idata/Readme_tnic3.txt",
)
FINRA_DAILY_BASE = "https://cdn.finra.org/equity/regsho/daily/"
DEFAULT_USER_AGENT = "agraemirolmez academic research public-data acquisition"


@dataclass(frozen=True)
class DownloadSpec:
    source: str
    url: str
    relative_path: str
    missing_is_expected: bool = False


@dataclass(frozen=True)
class DownloadRecord:
    source: str
    url: str
    relative_path: str
    byte_count: int
    sha256: str
    status: str


def _request(url: str, *, user_agent: str) -> urllib.request.Request:
    return urllib.request.Request(
        url,
        headers={
            "User-Agent": user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "identity",
        },
    )


def _read_url(url: str, *, user_agent: str) -> bytes:
    with urllib.request.urlopen(_request(url, user_agent=user_agent), timeout=60) as response:
        return response.read()


def _sec_specs(
    *, page_url: str, source: str, link_pattern: str, start_year: int, user_agent: str
) -> list[DownloadSpec]:
    html = _read_url(page_url, user_agent=user_agent).decode("utf-8", errors="replace")
    hrefs = re.findall(r'href=["\']([^"\']+)["\']', html, flags=re.IGNORECASE)
    specs: list[DownloadSpec] = []
    for href in hrefs:
        if not re.search(link_pattern, href, flags=re.IGNORECASE):
            continue
        filename = Path(urllib.parse.urlparse(href).path).name
        year_match = re.search(r"(20\d{2})", filename)
        if year_match is None or int(year_match.group(1)) < start_year:
            continue
        specs.append(
            DownloadSpec(
                source=source,
                url=urllib.parse.urljoin(page_url, href),
                relative_path=f"{source}/{filename}",
            )
        )
    unique = {spec.url: spec for spec in specs}
    if not unique:
        raise RuntimeError(f"no {source} links discovered from {page_url}")
    return sorted(unique.values(), key=lambda item: item.relative_path)


def _weekdays(start: dt.date, end: dt.date) -> Iterable[dt.date]:
    current = start
    while current <= end:
        if current.weekday() < 5:
            yield current
        current += dt.timedelta(days=1)


def _finra_specs(start: dt.date, end: dt.date) -> list[DownloadSpec]:
    specs: list[DownloadSpec] = []
    consolidated_start = dt.date(2018, 8, 1)
    for day in _weekdays(start, end):
        codes = ("CNMS",) if day >= consolidated_start else ("FNSQ", "FNYX", "FNRA")
        stamp = day.strftime("%Y%m%d")
        for code in codes:
            filename = f"{code}shvol{stamp}.txt"
            specs.append(
                DownloadSpec(
                    source="finra_short_volume",
                    url=urllib.parse.urljoin(FINRA_DAILY_BASE, filename),
                    relative_path=f"finra_short_volume/{day.year}/{filename}",
                    missing_is_expected=True,
                )
            )
    return specs


def _tnic_specs() -> list[DownloadSpec]:
    return [
        DownloadSpec(
            source="tnic",
            url=urllib.parse.urljoin(TNIC_BASE, relative),
            relative_path=f"tnic/{Path(relative).name}",
        )
        for relative in TNIC_FILES
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate(path: Path) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"empty download: {path}")
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            bad_member = archive.testzip()
            if bad_member is not None:
                raise RuntimeError(f"corrupt ZIP member {bad_member!r}: {path}")
            if not archive.namelist():
                raise RuntimeError(f"empty ZIP archive: {path}")
    elif "shvol" in path.name.lower():
        with path.open("rb") as handle:
            first_line = handle.readline(256).decode("ascii", errors="replace")
        if not first_line.startswith("Date|Symbol|ShortVolume"):
            raise RuntimeError(f"invalid FINRA header: {path}")


def _download_one(
    spec: DownloadSpec, *, root: Path, user_agent: str, retries: int = 3
) -> DownloadRecord | None:
    destination = root / spec.relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        try:
            _validate(destination)
            return DownloadRecord(
                source=spec.source,
                url=spec.url,
                relative_path=spec.relative_path,
                byte_count=destination.stat().st_size,
                sha256=_sha256(destination),
                status="reused",
            )
        except (OSError, RuntimeError, zipfile.BadZipFile):
            destination.unlink()

    partial = destination.with_name(destination.name + ".part")
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(
                _request(spec.url, user_agent=user_agent), timeout=120
            ) as response, partial.open("wb") as output:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
            _validate(partial)
            partial.replace(destination)
            return DownloadRecord(
                source=spec.source,
                url=spec.url,
                relative_path=spec.relative_path,
                byte_count=destination.stat().st_size,
                sha256=_sha256(destination),
                status="downloaded",
            )
        except urllib.error.HTTPError as exc:
            partial.unlink(missing_ok=True)
            if spec.missing_is_expected and exc.code in {403, 404}:
                return None
            if attempt == retries:
                raise
        except (OSError, RuntimeError, urllib.error.URLError, zipfile.BadZipFile):
            partial.unlink(missing_ok=True)
            if attempt == retries:
                raise
        time.sleep(float(attempt))
    raise AssertionError("unreachable")


def _write_manifest(root: Path, source: str, records: list[DownloadRecord]) -> Path:
    records = sorted(records, key=lambda item: item.relative_path)
    payload = {
        "schema": "research_v6_public_source_manifest_v1",
        "source": source,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "file_count": len(records),
        "total_bytes": sum(record.byte_count for record in records),
        "records": [asdict(record) for record in records],
    }
    destination = root / source / "manifest.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    partial.replace(destination)
    return destination


def _parse_date(value: str) -> dt.date:
    return dt.date.fromisoformat(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        action="append",
        choices=("sec_insiders", "sec_ftd", "finra_short_volume", "tnic"),
        help="Source to download; repeat as needed. Defaults to all sources.",
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--start-date", type=_parse_date, default=dt.date(2015, 1, 1))
    parser.add_argument("--end-date", type=_parse_date, default=dt.date.today())
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if args.end_date < args.start_date:
        parser.error("--end-date must not precede --start-date")
    if not 1 <= args.workers <= 16:
        parser.error("--workers must be between 1 and 16")

    selected = args.source or [
        "sec_insiders",
        "sec_ftd",
        "finra_short_volume",
        "tnic",
    ]
    user_agent = os.environ.get("SEC_USER_AGENT", DEFAULT_USER_AGENT)
    specs_by_source: dict[str, list[DownloadSpec]] = {}
    for source in selected:
        if source == "sec_insiders":
            specs_by_source[source] = _sec_specs(
                page_url=SEC_INSIDER_PAGE,
                source=source,
                link_pattern=r"form345\.zip$",
                start_year=args.start_date.year,
                user_agent=user_agent,
            )
        elif source == "sec_ftd":
            specs_by_source[source] = _sec_specs(
                page_url=SEC_FTD_PAGE,
                source=source,
                link_pattern=r"cnsfails.*\.zip$",
                start_year=args.start_date.year,
                user_agent=user_agent,
            )
        elif source == "finra_short_volume":
            specs_by_source[source] = _finra_specs(args.start_date, args.end_date)
        elif source == "tnic":
            specs_by_source[source] = _tnic_specs()

    if args.dry_run:
        for source, specs in specs_by_source.items():
            print(f"{source}: {len(specs)} candidate files")
        return 0

    args.root.mkdir(parents=True, exist_ok=True)
    for source, specs in specs_by_source.items():
        records: list[DownloadRecord] = []
        failures: list[str] = []
        print(f"{source}: processing {len(specs)} candidate files", flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_map = {
                executor.submit(
                    _download_one,
                    spec,
                    root=args.root,
                    user_agent=user_agent,
                ): spec
                for spec in specs
            }
            for completed, future in enumerate(
                concurrent.futures.as_completed(future_map), start=1
            ):
                spec = future_map[future]
                try:
                    record = future.result()
                    if record is not None:
                        records.append(record)
                except Exception as exc:  # continue independent file units
                    failures.append(f"{spec.url}: {type(exc).__name__}: {exc}")
                if completed % 50 == 0 or completed == len(specs):
                    _write_manifest(args.root, source, records)
                    print(
                        f"{source}: {completed}/{len(specs)} checked; "
                        f"{len(records)} present; {len(failures)} failed",
                        flush=True,
                    )
        manifest = _write_manifest(args.root, source, records)
        if failures:
            failure_path = args.root / source / "failures.json"
            failure_path.write_text(
                json.dumps(failures, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(f"{source}: failures recorded at {failure_path}", file=sys.stderr)
        print(f"{source}: manifest {manifest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
