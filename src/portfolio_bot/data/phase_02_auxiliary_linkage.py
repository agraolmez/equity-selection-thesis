"""Point-in-time auxiliary linkage onto the verified CRSP signal spine.

This module links identifiers only.  It does not select observations, engineer
features, construct labels, score securities, or form portfolios.
"""

from __future__ import annotations

import datetime as dt
import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import pyarrow as pa
import pyarrow.parquet as pq


class AuxiliaryLinkageError(RuntimeError):
    """Raised when row preservation or a frozen linkage rule is violated."""


LINKAGE_SCHEMA = pa.schema(
    [
        pa.field("signal_date", pa.date32(), nullable=False),
        pa.field("permno", pa.int64(), nullable=False),
        pa.field("ccm_match_status", pa.string(), nullable=False),
        pa.field("ccm_gvkey", pa.string()),
        pa.field("ccm_linktype", pa.string()),
        pa.field("ccm_linkprim", pa.string()),
        pa.field("sec_match_status", pa.string(), nullable=False),
        pa.field("sec_cik", pa.string()),
        pa.field("optionmetrics_match_status", pa.string(), nullable=False),
        pa.field("optionmetrics_secid", pa.int64()),
        pa.field("any_auxiliary_ambiguity", pa.bool_(), nullable=False),
        pa.field("universe_row_preserved", pa.bool_(), nullable=False),
    ]
)

ALLOWED_STATUSES = frozenset({"MATCHED", "UNMATCHED", "AMBIGUOUS", "NOT_COVERED"})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def _ccm_index(path: Path) -> dict[int, list[dict[str, Any]]]:
    columns = [
        "permno", "link_start_date", "link_end_date", "link_allowed",
        "link_precedence", "gvkey", "liid", "linktype", "linkprim", "cik",
    ]
    table = pq.read_table(path, columns=columns)
    result: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in table.to_pylist():
        if row["link_allowed"]:
            result[int(row["permno"])].append(row)
    for rows in result.values():
        rows.sort(
            key=lambda row: (
                row["link_start_date"],
                row["link_end_date"] or dt.date.max,
                row["link_precedence"] if row["link_precedence"] is not None else 32767,
                row["gvkey"],
                row["liid"],
            )
        )
    return dict(result)


def _option_index(path: Path) -> dict[int, list[dict[str, Any]]]:
    table = pq.read_table(
        path,
        columns=[
            "permno", "secid", "link_start_date", "link_end_date",
            "link_status", "candidate_count",
        ],
    )
    result: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in table.to_pylist():
        result[int(row["permno"])].append(row)
    for rows in result.values():
        rows.sort(key=lambda row: (row["link_start_date"], row["link_end_date"], row["secid"]))
    return dict(result)


def _active(rows: Iterable[Mapping[str, Any]], day: dt.date, start: str, end: str) -> list[Mapping[str, Any]]:
    return [
        row for row in rows
        if row[start] <= day and (row[end] is None or day <= row[end])
    ]


def resolve_ccm(rows: Iterable[Mapping[str, Any]], day: dt.date) -> dict[str, Any]:
    candidates = _active(rows, day, "link_start_date", "link_end_date")
    if not candidates:
        return {
            "ccm_match_status": "UNMATCHED", "ccm_gvkey": None,
            "ccm_linktype": None, "ccm_linkprim": None, "sec_match_status": "UNMATCHED",
            "sec_cik": None,
        }
    best_precedence = min(
        int(row["link_precedence"]) if row["link_precedence"] is not None else 32767
        for row in candidates
    )
    best = [
        row for row in candidates
        if (int(row["link_precedence"]) if row["link_precedence"] is not None else 32767)
        == best_precedence
    ]
    gvkeys = {str(row["gvkey"]) for row in best}
    if len(gvkeys) != 1:
        return {
            "ccm_match_status": "AMBIGUOUS", "ccm_gvkey": None,
            "ccm_linktype": None, "ccm_linkprim": None, "sec_match_status": "AMBIGUOUS",
            "sec_cik": None,
        }
    chosen = min(best, key=lambda row: (str(row["liid"]), str(row["linktype"])))
    # CIK is a company-level bridge, while LIID is security-level.  Resolve
    # the SEC bridge across all economically identical best-precedence rows;
    # otherwise a missing CIK on the lowest LIID can hide a valid company CIK,
    # or conflicting CIKs can be silently selected by an arbitrary LIID.
    ciks = {str(row["cik"]) for row in best if row.get("cik")}
    if len(ciks) > 1:
        sec_status = "AMBIGUOUS"
        cik = None
    elif ciks:
        sec_status = "MATCHED"
        cik = next(iter(ciks))
    else:
        sec_status = "UNMATCHED"
        cik = None
    return {
        "ccm_match_status": "MATCHED", "ccm_gvkey": str(chosen["gvkey"]),
        "ccm_linktype": str(chosen["linktype"]), "ccm_linkprim": str(chosen["linkprim"]),
        "sec_match_status": sec_status,
        "sec_cik": cik,
    }


def resolve_optionmetrics(rows: Iterable[Mapping[str, Any]], day: dt.date) -> dict[str, Any]:
    candidates = _active(rows, day, "link_start_date", "link_end_date")
    if not candidates:
        return {"optionmetrics_match_status": "NOT_COVERED", "optionmetrics_secid": None}
    secids = {int(row["secid"]) for row in candidates}
    ambiguous = any(row["link_status"] == "AMBIGUOUS" for row in candidates)
    if ambiguous or len(secids) != 1:
        return {"optionmetrics_match_status": "AMBIGUOUS", "optionmetrics_secid": None}
    if any(row["link_status"] != "MATCHED" for row in candidates):
        raise AuxiliaryLinkageError("Unexpected OptionMetrics link disposition")
    return {"optionmetrics_match_status": "MATCHED", "optionmetrics_secid": next(iter(secids))}


def build_auxiliary_signal_linkage(
    *,
    signal_universe_path: Path,
    ccm_link_path: Path,
    optionmetrics_link_path: Path,
    output_path: Path,
    batch_rows: int = 250_000,
) -> dict[str, Any]:
    """Write exactly one linkage row for every verified signal-universe row."""

    if batch_rows <= 0:
        raise AuxiliaryLinkageError("batch_rows must be positive")
    if output_path.exists():
        raise AuxiliaryLinkageError(f"Refusing to overwrite {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ccm = _ccm_index(ccm_link_path)
    options = _option_index(optionmetrics_link_path)
    source = pq.ParquetFile(signal_universe_path)
    writer = pq.ParquetWriter(output_path, LINKAGE_SCHEMA, compression="zstd", use_dictionary=True)
    pending: list[dict[str, Any]] = []
    source_rows = output_rows = 0
    previous: tuple[dt.date, int] | None = None
    status_counts: Counter[str] = Counter()
    ambiguity_rows = 0
    try:
        for batch in source.iter_batches(
            batch_size=batch_rows,
            columns=["signal_date", "permno"],
        ):
            for signal_date, permno in zip(
                batch.column(0).to_pylist(), batch.column(1).to_pylist(), strict=True
            ):
                source_rows += 1
                key = (signal_date, int(permno))
                if previous is not None and key <= previous:
                    raise AuxiliaryLinkageError("Signal spine is not strictly sorted and unique")
                previous = key
                ccm_result = resolve_ccm(ccm.get(int(permno), ()), signal_date)
                option_result = resolve_optionmetrics(options.get(int(permno), ()), signal_date)
                ambiguity = (
                    ccm_result["ccm_match_status"] == "AMBIGUOUS"
                    or ccm_result["sec_match_status"] == "AMBIGUOUS"
                    or option_result["optionmetrics_match_status"] == "AMBIGUOUS"
                )
                pending.append(
                    {
                        "signal_date": signal_date,
                        "permno": int(permno),
                        **ccm_result,
                        **option_result,
                        "any_auxiliary_ambiguity": ambiguity,
                        "universe_row_preserved": True,
                    }
                )
                for family, status in (
                    ("ccm", ccm_result["ccm_match_status"]),
                    ("sec", ccm_result["sec_match_status"]),
                    ("optionmetrics", option_result["optionmetrics_match_status"]),
                ):
                    if status not in ALLOWED_STATUSES:
                        raise AuxiliaryLinkageError(f"Invalid {family} status {status}")
                    status_counts[f"{family}:{status}"] += 1
                ambiguity_rows += int(ambiguity)
                if len(pending) >= batch_rows:
                    writer.write_table(pa.Table.from_pylist(pending, schema=LINKAGE_SCHEMA))
                    output_rows += len(pending)
                    pending.clear()
        if pending:
            writer.write_table(pa.Table.from_pylist(pending, schema=LINKAGE_SCHEMA))
            output_rows += len(pending)
        writer.close()
    except BaseException:
        # The destination did not exist before this call, so a failed build may
        # safely remove only its own incomplete Parquet.  Leaving it behind
        # would make the immutable-output guard turn a recoverable failure into
        # a manual-cleanup requirement on the next attempt.
        try:
            writer.close()
        finally:
            output_path.unlink(missing_ok=True)
        raise
    if source_rows != output_rows:
        raise AuxiliaryLinkageError(
            f"Signal spine row preservation failed: {source_rows} != {output_rows}"
        )
    return {
        "table": "auxiliary_signal_linkage",
        "path": output_path.name,
        "rows": output_rows,
        "source_spine_rows": source_rows,
        "primary_key_preserved": True,
        "all_universe_rows_preserved": True,
        "ambiguity_rows": ambiguity_rows,
        "status_counts": dict(sorted(status_counts.items())),
        "size_bytes": output_path.stat().st_size,
        "sha256": _sha256(output_path),
    }
