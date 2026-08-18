"""Build Compustat/CCM and factor/liquidity Stage 2 staging tables."""

from __future__ import annotations

import csv
import datetime as dt
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

from portfolio_bot.data.phase_02_auxiliary_core import (
    AuxiliaryBuildError,
    TradingCalendar,
    normalize_cusip8,
    optional_date,
    optional_decimal,
    optional_int,
    optional_text,
    required_date,
    required_int,
    required_text,
    sha256,
    source_record_id,
    table_contract,
    table_from_rows,
    table_schema,
)


LAST_SIGNAL = dt.date(2014, 10, 31)
LAST_EXIT = dt.date(2014, 12, 1)


class _Sink:
    def __init__(
        self,
        schema_document: Mapping[str, Any],
        name: str,
        output_dir: Path,
        *,
        batch_rows: int = 25_000,
    ) -> None:
        self.name = name
        self.schema = table_schema(schema_document, name)
        self.path = output_dir / f"{name}.parquet"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.writer = pq.ParquetWriter(
            self.path, self.schema, compression="zstd", use_dictionary=True
        )
        self.batch_rows = batch_rows
        self.rows: list[Mapping[str, Any]] = []
        self.count = 0

    def add(self, row: Mapping[str, Any]) -> None:
        self.rows.append(row)
        if len(self.rows) >= self.batch_rows:
            self.flush()

    def flush(self) -> None:
        if not self.rows:
            return
        table = pa.Table.from_pylist(self.rows, schema=self.schema)
        self.writer.write_table(table, row_group_size=self.batch_rows)
        self.count += table.num_rows
        self.rows.clear()

    def close(self) -> dict[str, Any]:
        self.flush()
        self.writer.close()
        parquet = pq.ParquetFile(self.path)
        if not parquet.schema_arrow.equals(self.schema, check_metadata=True):
            raise AuxiliaryBuildError(f"Arrow schema differs for {self.name}")
        if parquet.metadata.num_rows != self.count:
            raise AuxiliaryBuildError(f"Row count differs for {self.name}")
        return {
            "table": self.name,
            "rows": self.count,
            "path": self.path.name,
            "sha256": sha256(self.path),
            "size_bytes": self.path.stat().st_size,
        }


def _open_filing_index(path: Path | None) -> dict[str, list[tuple[dt.date, dt.date, str]]]:
    if path is None:
        return {}
    table = pq.read_table(
        path,
        columns=["cik", "filing_date", "form_family", "public_availability_date"],
    )
    index: dict[str, list[tuple[dt.date, dt.date, str]]] = defaultdict(list)
    for cik, filing, form, available in zip(
        table["cik"].to_pylist(),
        table["filing_date"].to_pylist(),
        table["form_family"].to_pylist(),
        table["public_availability_date"].to_pylist(),
        strict=True,
    ):
        if form in {"10-K", "20-F", "40-F"}:
            index[str(cik)].append((filing, available, str(form)))
    for values in index.values():
        values.sort()
    return dict(index)


def build_ccm_link_intervals(
    *,
    raw_path: Path,
    output_dir: Path,
    schema_document: Mapping[str, Any],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    with raw_path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            start = required_date(raw["LINKDT"], "LINKDT")
            end = optional_date(raw["LINKENDDT"], "LINKENDDT")
            if start > LAST_SIGNAL or (end is not None and end < dt.date(2000, 1, 1)):
                continue
            linktype = required_text(raw["LINKTYPE"], "LINKTYPE")
            linkprim = required_text(raw["LINKPRIM"], "LINKPRIM")
            allowed = linktype in {"LC", "LU"} and linkprim in {"P", "C"}
            rows.append(
                {
                    "gvkey": required_text(raw["gvkey"], "gvkey"),
                    "liid": required_text(raw["LIID"], "LIID"),
                    "permno": required_int(raw["LPERMNO"], "LPERMNO"),
                    "permco": optional_int(raw["LPERMCO"], "LPERMCO"),
                    "link_start_date": start,
                    "link_end_date": end,
                    "linktype": linktype,
                    "linkprim": linkprim,
                    "link_allowed": allowed,
                    "link_precedence": (1 if linkprim == "P" else 2 if linkprim == "C" else None),
                    "normalized_cusip8": normalize_cusip8(raw.get("cusip")),
                    "cik": optional_text(raw.get("cik")),
                    "fic": optional_text(raw.get("FIC")),
                    "loc": optional_text(raw.get("LOC")),
                    "incorp": optional_text(raw.get("INCORP")),
                }
            )
    rows.sort(
        key=lambda row: (
            row["permno"], row["link_start_date"], row["gvkey"], row["liid"],
            row["link_end_date"] or dt.date.max, row["linktype"], row["linkprim"],
        )
    )
    table = table_from_rows(schema_document, "ccm_link_interval", rows)
    path = output_dir / "ccm_link_interval.parquet"
    output_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd", use_dictionary=True)
    return {
        "table": "ccm_link_interval",
        "rows": table.num_rows,
        "allowed_rows": sum(row["link_allowed"] for row in rows),
        "path": path.name,
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _annual_availability(
    *,
    datadate: dt.date,
    cik: str | None,
    calendar: TradingCalendar,
    filings: Mapping[str, list[tuple[dt.date, dt.date, str]]],
) -> tuple[dt.date, str] | None:
    candidates = [
        value
        for value in filings.get(cik or "", [])
        if datadate < value[0] <= datadate + dt.timedelta(days=180)
    ]
    if candidates:
        first_date = candidates[0][0]
        earliest = [value for value in candidates if value[0] == first_date]
        available = min(value[1] for value in earliest)
        return available, "matched_10k_20f_40f_filing_plus_one_trading_day"
    fallback_target = datadate + dt.timedelta(days=180)
    if fallback_target > LAST_SIGNAL:
        return None
    fallback = calendar.on_or_after(fallback_target)
    return fallback, "datadate_plus_180_calendar_days_fallback"


def _fundamental_row(
    *,
    name: str,
    filename: str,
    raw: Mapping[str, str],
    ordinal: int,
    calendar: TradingCalendar,
    filings: Mapping[str, list[tuple[dt.date, dt.date, str]]],
    contract: Mapping[str, Any],
) -> dict[str, Any] | None:
    gvkey = required_text(raw["GVKEY"], "GVKEY")
    datadate = required_date(raw["datadate"], "datadate")
    # The date gate is deliberately evaluated before any post-boundary payload
    # conversion.  Development staging may inspect the availability-driving
    # date fields, but it may not decode validation/final predictor payloads.
    if datadate > LAST_SIGNAL:
        return None
    cik = optional_text(raw.get("cik"))
    if name == "compustat_annual_observation":
        availability = _annual_availability(
            datadate=datadate, cik=cik, calendar=calendar, filings=filings
        )
        if availability is None:
            return None
        available, basis = availability
        stale_days = 540
    elif name == "compustat_quarterly_observation":
        rdq = optional_date(raw.get("rdq"), "rdq")
        if rdq is not None:
            if rdq >= LAST_SIGNAL:
                return None
            available = calendar.strictly_after(rdq)
            basis = "rdq_plus_one_verified_trading_day"
        else:
            fallback_target = datadate + dt.timedelta(days=120)
            if fallback_target > LAST_SIGNAL:
                return None
            available = calendar.on_or_after(fallback_target)
            basis = "datadate_plus_120_calendar_days_fallback"
        stale_days = 180
    else:
        srcdate = optional_date(raw.get("srcdate"), "srcdate")
        if srcdate is not None:
            if srcdate >= LAST_SIGNAL:
                return None
            available = calendar.strictly_after(srcdate)
            basis = "srcdate_plus_one_verified_trading_day"
        else:
            fallback_target = datadate + dt.timedelta(days=180)
            if fallback_target > LAST_SIGNAL:
                return None
            available = calendar.on_or_after(fallback_target)
            basis = "datadate_plus_180_calendar_days_fallback"
        stale_days = 540
    if available > LAST_SIGNAL:
        return None
    link_start = optional_date(raw.get("LINKDT"), "LINKDT")
    link_end = optional_date(raw.get("LINKENDDT"), "LINKENDDT")
    record = source_record_id(
        filename,
        ordinal,
        [gvkey, datadate, raw.get("LPERMNO"), raw.get("LINKDT"), raw.get("LINKENDDT")],
    )
    if name == "compustat_annual_observation":
        row: dict[str, Any] = {
            "source_record_id": record, "gvkey": gvkey, "datadate": datadate,
            "permno": optional_int(raw.get("LPERMNO"), "LPERMNO"),
            "link_start_date": link_start, "link_end_date": link_end,
            "indfmt": required_text(raw.get("indfmt"), "indfmt"),
            "consol": required_text(raw.get("consol"), "consol"),
            "popsrc": required_text(raw.get("popsrc"), "popsrc"),
            "datafmt": required_text(raw.get("datafmt"), "datafmt"),
            "standard_format_eligible": all(
                raw.get(key) == value
                for key, value in {"indfmt":"INDL","consol":"C","popsrc":"D","datafmt":"STD"}.items()
            ),
            "currency_code": optional_text(raw.get("curcd")), "cik": cik,
            "fic": optional_text(raw.get("fic")), "loc": optional_text(raw.get("loc")),
            "observation_date": datadate, "public_availability_date": available,
            "availability_basis": basis,
            "stale_after_date": available + dt.timedelta(days=stale_days),
        }
    elif name == "compustat_quarterly_observation":
        row = {
            "source_record_id": record, "gvkey": gvkey, "datadate": datadate,
            "rdq": optional_date(raw.get("rdq"), "rdq"),
            "permno": optional_int(raw.get("LPERMNO"), "LPERMNO"),
            "link_start_date": link_start, "link_end_date": link_end,
            "indfmt": required_text(raw.get("indfmt"), "indfmt"),
            "consol": required_text(raw.get("consol"), "consol"),
            "popsrc": required_text(raw.get("popsrc"), "popsrc"),
            "datafmt": required_text(raw.get("datafmt"), "datafmt"),
            "standard_format_eligible": all(
                raw.get(key) == value
                for key, value in {"indfmt":"INDL","consol":"C","popsrc":"D","datafmt":"STD"}.items()
            ),
            "currency_code": optional_text(raw.get("curcdq")), "cik": cik,
            "fic": optional_text(raw.get("fic")), "loc": optional_text(raw.get("loc")),
            "observation_date": datadate, "public_availability_date": available,
            "availability_basis": basis,
            "stale_after_date": available + dt.timedelta(days=stale_days),
        }
    else:
        row = {
            "source_record_id": record, "gvkey": gvkey,
            "liid": optional_text(raw.get("LIID")), "datadate": datadate,
            "srcdate": optional_date(raw.get("srcdate"), "srcdate"),
            "stype": required_text(raw.get("stype"), "stype"),
            "sid": required_text(raw.get("sid"), "sid"),
            "permno": optional_int(raw.get("LPERMNO"), "LPERMNO"),
            "link_start_date": link_start, "link_end_date": link_end,
            "currency_code": optional_text(raw.get("curcds")),
            "geotp": optional_text(raw.get("geotp")),
            "segment_name": optional_text(raw.get("snms")),
            "observation_date": datadate, "public_availability_date": available,
            "availability_basis": basis,
            "stale_after_date": available + dt.timedelta(days=stale_days),
            "sales": optional_decimal(raw.get("sales"), scale=6, label="sales"),
            "revenue": optional_decimal(raw.get("revts"), scale=6, label="revts"),
            "assets": optional_decimal(raw.get("atlls"), scale=6, label="atlls"),
        }
    for payload in contract.get("source_payload_columns", []):
        if payload in {"fyear", "fyearq", "fqtr", "fyr"}:
            row[payload] = optional_int(raw.get(payload), payload)
        elif payload == "datafqtr":
            row[payload] = optional_text(raw.get(payload))
        else:
            row[payload] = optional_decimal(raw.get(payload), scale=12, label=payload)
    return row


def build_fundamental_table(
    *,
    name: str,
    raw_path: Path,
    output_dir: Path,
    schema_document: Mapping[str, Any],
    calendar: TradingCalendar,
    sec_filing_path: Path | None = None,
) -> dict[str, Any]:
    if name not in {
        "compustat_annual_observation", "compustat_quarterly_observation",
        "compustat_segment_observation",
    }:
        raise AuxiliaryBuildError(f"Unsupported fundamental table {name}")
    filings = _open_filing_index(sec_filing_path) if name == "compustat_annual_observation" else {}
    contract = table_contract(schema_document, name)
    sink = _Sink(schema_document, name, output_dir)
    group_key: tuple[str, dt.date] | None = None
    group_rows: list[dict[str, Any]] = []
    prior_group: tuple[str, dt.date] | None = None

    def flush_group() -> None:
        nonlocal group_rows
        for item in sorted(group_rows, key=lambda value: value["source_record_id"]):
            sink.add(item)
        group_rows = []

    with raw_path.open(newline="", encoding="utf-8") as handle:
        for ordinal, raw in enumerate(csv.DictReader(handle), start=1):
            row = _fundamental_row(
                name=name, filename=raw_path.name, raw=raw, ordinal=ordinal,
                calendar=calendar, filings=filings, contract=contract,
            )
            if row is None:
                continue
            current = (row["gvkey"], row["datadate"])
            if prior_group is not None and current < prior_group:
                raise AuxiliaryBuildError(f"Raw order is not compatible with frozen sort for {name}")
            if group_key is not None and current != group_key:
                flush_group()
            group_key = current
            prior_group = current
            group_rows.append(row)
    flush_group()
    return sink.close()


def build_factor_tables(
    *,
    raw_root: Path,
    output_dir: Path,
    schema_document: Mapping[str, Any],
    calendar: TradingCalendar,
) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    factor_columns = ("mktrf", "smb", "hml", "rmw", "cma", "rf", "umd")
    for name, filename, date_column, end_date in (
        ("ff_factor_daily_observation", "ff_5_momentum_daily.csv", "date", LAST_EXIT),
        ("ff_factor_monthly_observation", "ff_5_momentum_monthly.csv", "dateff", LAST_EXIT),
    ):
        rows: list[dict[str, Any]] = []
        with (raw_root / filename).open(newline="", encoding="utf-8") as handle:
            for raw in csv.DictReader(handle):
                observation = required_date(raw[date_column], date_column)
                if observation > end_date:
                    continue
                row: dict[str, Any] = {
                    "observation_date": observation,
                    "public_availability_date": observation,
                    "availability_basis": "dated_factor_close",
                }
                if name == "ff_factor_monthly_observation":
                    row["factor_month"] = observation
                for column in factor_columns:
                    value = optional_decimal(raw[column], scale=12, label=column)
                    if value is None:
                        raise AuxiliaryBuildError(f"Blank required factor {column}")
                    row[column] = value
                rows.append(row)
        rows.sort(key=lambda row: row.get("factor_month", row["observation_date"]))
        table = table_from_rows(schema_document, name, rows)
        path = output_dir / f"{name}.parquet"
        output_dir.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, path, compression="zstd", use_dictionary=True)
        outputs.append({
            "table": name, "rows": table.num_rows, "path": path.name,
            "sha256": sha256(path), "size_bytes": path.stat().st_size,
        })

    ps_rows: list[dict[str, Any]] = []
    with (raw_root / "ps_liquidity_monthly.csv").open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            observation = required_date(raw["DATE"], "DATE")
            following_month = (observation.replace(day=1) + dt.timedelta(days=63)).replace(day=1)
            one_full_month_later = following_month - dt.timedelta(days=1)
            available = calendar.on_or_after(one_full_month_later)
            if available > LAST_SIGNAL:
                continue
            ps_rows.append({
                "observation_month": observation,
                "observation_date": observation,
                "recorded_release_date": None,
                "public_availability_date": available,
                "availability_basis": "one_full_calendar_month_lag_fallback",
                "ps_level": optional_decimal(raw.get("PS_LEVEL"), scale=12, label="PS_LEVEL"),
                "ps_vwf_audit_only": optional_decimal(raw.get("PS_VWF"), scale=12, label="PS_VWF"),
            })
    ps_rows.sort(key=lambda row: row["observation_month"])
    ps_table = table_from_rows(schema_document, "ps_liquidity_observation", ps_rows)
    ps_path = output_dir / "ps_liquidity_observation.parquet"
    pq.write_table(ps_table, ps_path, compression="zstd", use_dictionary=True)
    outputs.append({
        "table": "ps_liquidity_observation", "rows": ps_table.num_rows,
        "path": ps_path.name, "sha256": sha256(ps_path),
        "size_bytes": ps_path.stat().st_size,
    })
    return outputs


def scan_ccm_market_crosscheck(
    *,
    raw_path: Path,
    valid_crsp_permnos: pa.Array,
) -> dict[str, Any]:
    """Fully scan one redundant CCM market file without materializing rows."""

    columns = ["datadate", "LPERMNO", "LINKDT", "LINKENDDT", "LINKTYPE", "LINKPRIM"]
    reader = pacsv.open_csv(
        raw_path,
        read_options=pacsv.ReadOptions(block_size=16 << 20, use_threads=True),
        convert_options=pacsv.ConvertOptions(
            include_columns=columns,
            column_types={name: pa.string() for name in columns},
            strings_can_be_null=False,
        ),
    )
    totals: Counter[str] = Counter()
    minimum: dt.date | None = None
    maximum: dt.date | None = None
    cutoff = pa.scalar(LAST_EXIT, type=pa.date32())
    start_scope = pa.scalar(dt.date(2000, 1, 1), type=pa.date32())
    for batch_number, batch in enumerate(reader):
        table = pa.Table.from_batches([batch])
        dates = pc.cast(table["datadate"], pa.date32())
        in_scope = pc.and_(pc.greater_equal(dates, start_scope), pc.less_equal(dates, cutoff))
        scoped = table.filter(in_scope)
        scoped_dates = pc.cast(scoped["datadate"], pa.date32())
        totals["raw_rows"] += table.num_rows
        totals["development_rows"] += scoped.num_rows
        if scoped.num_rows:
            batch_min = pc.min(scoped_dates).as_py()
            batch_max = pc.max(scoped_dates).as_py()
            minimum = batch_min if minimum is None else min(minimum, batch_min)
            maximum = batch_max if maximum is None else max(maximum, batch_max)
            permno = pc.cast(scoped["LPERMNO"], pa.int64())
            starts = pc.cast(scoped["LINKDT"], pa.date32())
            ends_text = scoped["LINKENDDT"]
            ends = pc.cast(
                pc.if_else(
                    pc.equal(ends_text, "E"),
                    pa.scalar(None, type=pa.string()),
                    ends_text,
                ),
                pa.date32(),
            )
            totals["crsp_permno_not_found"] += int(
                pc.sum(pc.cast(pc.invert(pc.is_in(permno, value_set=valid_crsp_permnos)), pa.int64())).as_py()
            )
            interval_ok = pc.and_kleene(
                pc.less_equal(starts, scoped_dates),
                pc.or_kleene(pc.is_null(ends), pc.less_equal(scoped_dates, ends)),
            )
            totals["link_interval_violations"] += int(
                pc.sum(pc.cast(pc.invert(interval_ok), pa.int64())).as_py()
            )
            allowed = pc.and_(
                pc.is_in(scoped["LINKTYPE"], value_set=pa.array(["LC", "LU"])),
                pc.is_in(scoped["LINKPRIM"], value_set=pa.array(["P", "C"])),
            )
            totals["allowed_link_rows"] += int(pc.sum(pc.cast(allowed, pa.int64())).as_py())
        if batch_number and batch_number % 50 == 0:
            print(f"CCM QA scan {raw_path.name}: {totals['raw_rows']:,} rows", flush=True)
    return {
        "source": raw_path.name,
        **dict(totals),
        "development_min_date": None if minimum is None else minimum.isoformat(),
        "development_max_date": None if maximum is None else maximum.isoformat(),
        "row_level_materialized": False,
        "source_sha256": sha256(raw_path),
    }


def build_family(
    *,
    raw_root: Path,
    output_dir: Path,
    schema_document: Mapping[str, Any],
    calendar: TradingCalendar,
    sec_filing_path: Path | None = None,
) -> dict[str, Any]:
    outputs = [
        build_ccm_link_intervals(
            raw_path=raw_root / "ccm_link_table.csv", output_dir=output_dir,
            schema_document=schema_document,
        )
    ]
    for name, filename in (
        ("compustat_annual_observation", "comp_funda_ccm.csv"),
        ("compustat_quarterly_observation", "comp_fundq_ccm.csv"),
        ("compustat_segment_observation", "comp_segments_ccm.csv"),
    ):
        print(f"auxiliary Compustat: building {name}", flush=True)
        outputs.append(
            build_fundamental_table(
                name=name, raw_path=raw_root / filename, output_dir=output_dir,
                schema_document=schema_document, calendar=calendar,
                sec_filing_path=sec_filing_path,
            )
        )
    outputs.extend(
        build_factor_tables(
            raw_root=raw_root, output_dir=output_dir,
            schema_document=schema_document, calendar=calendar,
        )
    )
    return {
        "family": "compustat_ccm_factors_liquidity",
        "outputs": outputs,
        "output_rows": sum(item["rows"] for item in outputs),
    }
