"""Point-in-time ownership, filing, event, segment, and identity features."""

from __future__ import annotations

import datetime as dt
import math
import re
import unicodedata
from collections import defaultdict
from typing import Any, Iterable, Mapping

from portfolio_bot.features.feature_lineage import (
    FeatureFamilyResult,
    FeatureLineage,
    make_lineage,
)
from portfolio_bot.features.feature_store_contract import classify_ff12


FEATURE_NAMES = (
    "institutional_ownership_ratio",
    "institutional_ownership_change_1q",
    "institutional_holder_breadth",
    "institutional_ownership_hhi",
    "days_since_latest_annual_filing",
    "days_since_latest_interim_financial_filing",
    "latest_annual_filing_form",
    "latest_interim_financial_filing_form",
    "count_8k_30d",
    "count_8k_90d",
    "count_6k_30d",
    "count_6k_90d",
    "recency_8k_30d",
    "recency_8k_90d",
    "recency_6k_30d",
    "recency_6k_90d",
    "restatement_disclosed_90d",
    "restatement_disclosed_365d",
    "business_segment_count",
    "geographic_segment_count",
    "business_segment_revenue_hhi",
    "business_segment_revenue_hhi_change_1y",
    "eight_k_item_families_30d",
    "eight_k_item_families_90d",
    "foreign_listing_indicator",
    "adr_indicator",
    "us_incorporation_indicator",
    "exchange_category",
    "ff12_sector_category",
    "sic2_industry_category",
    "domicile_country_category",
    "reporting_currency_category",
)

ANNUAL_FORMS = frozenset({"10-K", "20-F", "40-F"})
INTERIM_FORMS = frozenset({"10-Q"})
COUNTRY_PATTERN = re.compile(r"^[A-Z]{3}$")
SIC_PATTERN = re.compile(r"^[0-9]{1,4}$")


class OwnershipEventFeatureError(RuntimeError):
    """Raised for ambiguous authoritative filing, identity, or shares grains."""


def _number(value: Any, *, nonnegative: bool = False) -> float | None:
    if value is None:
        return None
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        return None
    return result


def normalize_text(value: Any) -> str | None:
    if value is None:
        return None
    result = unicodedata.normalize("NFKC", str(value)).strip().upper()
    return result or None


def normalize_8k_item(value: Any) -> str | None:
    return normalize_text(value)


def _normalize_cusip8(value: Any) -> str | None:
    normalized = normalize_text(value)
    if normalized is None:
        return None
    compact = "".join(character for character in normalized if character.isalnum())
    return compact[:8] if len(compact) >= 8 else None


def _known_country(value: Any) -> str | None:
    normalized = normalize_text(value)
    return normalized if normalized is not None and COUNTRY_PATTERN.fullmatch(normalized) else None


def _eligible(row: Mapping[str, Any], signal_date: dt.date) -> bool:
    observation = row.get("observation_date")
    availability = row.get("public_availability_date")
    return (
        isinstance(observation, dt.date)
        and isinstance(availability, dt.date)
        and observation <= signal_date
        and availability <= signal_date
    )


def compute_identity_features(
    identity_row: Mapping[str, Any] | None,
    *,
    fundamental_fic: str | None,
    fundamental_loc: str | None,
    reporting_currency: str | None,
    ff12_mapping: Mapping[str, Any],
) -> dict[str, float | int | str | None]:
    """Resolve the eight frozen identity concepts from availability-aligned inputs."""

    identity = identity_row or {}
    share_type = normalize_text(identity.get("share_type"))
    us_flag = normalize_text(identity.get("us_incorporation_flag"))
    exchange = normalize_text(identity.get("primary_exchange"))
    exchange = exchange if exchange in {"N", "A", "Q"} else "UNKNOWN"

    sic = normalize_text(identity.get("sic_code"))
    normalized_sic = sic.zfill(4) if sic is not None and SIC_PATTERN.fullmatch(sic) else None
    sic2 = normalized_sic[:2] if normalized_sic is not None else "UNKNOWN_INDUSTRY"
    ff12 = classify_ff12(normalized_sic, ff12_mapping)

    fic = _known_country(fundamental_fic)
    loc = _known_country(fundamental_loc)
    if fic is not None:
        domicile = fic
    elif loc is not None:
        domicile = loc
    elif us_flag == "Y":
        domicile = "USA"
    elif us_flag == "N":
        domicile = "UNKNOWN_FOREIGN"
    else:
        domicile = "UNKNOWN_COUNTRY"
    foreign = None if domicile == "UNKNOWN_COUNTRY" else int(domicile != "USA")

    currency = normalize_text(reporting_currency) or "UNKNOWN_CURRENCY"
    return {
        "foreign_listing_indicator": foreign,
        "adr_indicator": 1 if share_type == "AD" else (0 if share_type == "NS" else None),
        "us_incorporation_indicator": 1 if us_flag == "Y" else (0 if us_flag == "N" else None),
        "exchange_category": exchange,
        "ff12_sector_category": ff12,
        "sic2_industry_category": sic2,
        "domicile_country_category": domicile,
        "reporting_currency_category": currency,
    }


def _latest_filing(
    rows: Iterable[Mapping[str, Any]], forms: frozenset[str]
) -> Mapping[str, Any] | None:
    eligible = [row for row in rows if normalize_text(row.get("form_family")) in forms]
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda row: (
            row["public_availability_date"],
            row.get("filing_date") or dt.date.min,
            str(row.get("source_record_id", "")),
        ),
    )


def _window_rows(
    rows: Iterable[Mapping[str, Any]], signal_date: dt.date, days: int
) -> list[Mapping[str, Any]]:
    lower = signal_date - dt.timedelta(days=days)
    return [row for row in rows if lower < row["public_availability_date"] <= signal_date]


def compute_sec_event_features(
    *,
    sec_match_status: str,
    cik: str | None,
    signal_date: dt.date,
    filing_rows: Iterable[Mapping[str, Any]],
    item_rows: Iterable[Mapping[str, Any]],
) -> dict[str, float | int | str | list[str] | None]:
    names = (
        "days_since_latest_annual_filing",
        "days_since_latest_interim_financial_filing",
        "latest_annual_filing_form",
        "latest_interim_financial_filing_form",
        "count_8k_30d", "count_8k_90d", "count_6k_30d", "count_6k_90d",
        "recency_8k_30d", "recency_8k_90d", "recency_6k_30d", "recency_6k_90d",
        "eight_k_item_families_30d", "eight_k_item_families_90d",
    )
    if sec_match_status != "MATCHED" or cik is None:
        values: dict[str, float | int | str | list[str] | None] = {name: None for name in names}
        values["latest_annual_filing_form"] = "UNKNOWN"
        values["latest_interim_financial_filing_form"] = "UNKNOWN"
        values["eight_k_item_families_30d"] = ["UNKNOWN"]
        values["eight_k_item_families_90d"] = ["UNKNOWN"]
        return values

    normalized_cik = normalize_text(cik)
    filings = [
        row for row in filing_rows
        if normalize_text(row.get("cik")) == normalized_cik and _eligible(row, signal_date)
    ]
    source_ids = [str(row.get("source_record_id", "")) for row in filings]
    if len(source_ids) != len(set(source_ids)):
        raise OwnershipEventFeatureError("Duplicate eligible SEC filing source_record_id")
    annual = _latest_filing(filings, ANNUAL_FORMS)
    interim = _latest_filing(filings, INTERIM_FORMS)
    result: dict[str, float | int | str | list[str] | None] = {
        "days_since_latest_annual_filing": None if annual is None else (signal_date - annual["public_availability_date"]).days,
        "days_since_latest_interim_financial_filing": None if interim is None else (signal_date - interim["public_availability_date"]).days,
        "latest_annual_filing_form": "UNKNOWN" if annual is None else normalize_text(annual["form_family"]),
        "latest_interim_financial_filing_form": "UNKNOWN" if interim is None else "10-Q",
    }
    for form, stem in (("8-K", "8k"), ("6-K", "6k")):
        form_rows = [row for row in filings if normalize_text(row.get("form_family")) == form]
        for days in (30, 90):
            window = _window_rows(form_rows, signal_date, days)
            distinct = {str(row["source_record_id"]): row for row in window}
            result[f"count_{stem}_{days}d"] = len(distinct)
            result[f"recency_{stem}_{days}d"] = (
                None
                if not distinct
                else (signal_date - max(row["public_availability_date"] for row in distinct.values())).days
            )

    items = [
        row for row in item_rows
        if normalize_text(row.get("cik")) == normalized_cik
        and normalize_text(row.get("form_family")) == "8-K"
        and _eligible(row, signal_date)
    ]
    item_ids = [str(row.get("source_record_id", "")) for row in items]
    if len(item_ids) != len(set(item_ids)):
        raise OwnershipEventFeatureError("Duplicate eligible 8-K item source_record_id")
    for days in (30, 90):
        categories = sorted(
            {
                normalized
                for row in _window_rows(items, signal_date, days)
                if (normalized := normalize_8k_item(row.get("item_number"))) is not None
            },
            key=lambda value: value.encode("utf-8"),
        )
        result[f"eight_k_item_families_{days}d"] = categories or ["NO_EVENT"]
    return result


def compute_restatement_features(
    *,
    positively_covered: bool,
    signal_date: dt.date,
    event_rows: Iterable[Mapping[str, Any]],
) -> dict[str, int | None]:
    if not positively_covered:
        return {"restatement_disclosed_90d": None, "restatement_disclosed_365d": None}
    events = [row for row in event_rows if _eligible(row, signal_date)]
    return {
        f"restatement_disclosed_{days}d": int(bool(_window_rows(events, signal_date, days)))
        for days in (90, 365)
    }


def _active(rows: Iterable[Mapping[str, Any]], day: dt.date) -> list[Mapping[str, Any]]:
    return [
        row for row in rows
        if row.get("start_date") <= day and (row.get("end_date") is None or day <= row["end_date"])
    ]


def _target_position(
    filing: Mapping[str, Any],
    *,
    target_permno: int,
    holdings_by_filing: Mapping[str, list[Mapping[str, Any]]],
    identity_intervals: Iterable[Mapping[str, Any]],
    shares_intervals: Iterable[Mapping[str, Any]],
) -> tuple[float, float | None] | None:
    report_date = filing["report_date"]
    identity_by_cusip: dict[str, set[int]] = defaultdict(set)
    for row in _active(identity_intervals, report_date):
        cusip = _normalize_cusip8(row.get("cusip"))
        if cusip is not None:
            identity_by_cusip[cusip].add(int(row["permno"]))
    reported_shares = 0.0
    for row in holdings_by_filing.get(str(filing["filing_id"]), []):
        if normalize_text(row.get("put_call")) is not None:
            continue
        cusip = _normalize_cusip8(row.get("normalized_cusip8"))
        linked = None if cusip is None else identity_by_cusip.get(cusip)
        if linked != {target_permno}:
            continue
        shares = _number(row.get("reported_shares"), nonnegative=True)
        if shares is None:
            return None
        reported_shares += shares
    if reported_shares == 0.0:
        return 0.0, 0.0
    denominator_rows = [
        row for row in _active(shares_intervals, report_date) if int(row["permno"]) == target_permno
    ]
    if len(denominator_rows) != 1:
        return reported_shares, None
    denominator = _number(denominator_rows[0].get("shares_outstanding"), nonnegative=True)
    if denominator is None or denominator <= 0.0:
        return reported_shares, None
    return reported_shares, reported_shares / denominator


def _select_13f_snapshots(
    filing_rows: Iterable[Mapping[str, Any]], signal_date: dt.date
) -> tuple[
    list[Mapping[str, Any]],
    dict[str, tuple[Mapping[str, Any], Mapping[str, Any] | None]],
]:
    all_filings = list(filing_rows)
    eligible = [row for row in all_filings if _eligible(row, signal_date)]
    filing_ids = [str(row["filing_id"]) for row in eligible]
    if len(filing_ids) != len(set(filing_ids)):
        raise OwnershipEventFeatureError("Duplicate eligible 13F filing_id")

    folded: dict[tuple[str, dt.date], Mapping[str, Any]] = {}
    for row in eligible:
        key = (str(row["manager_cik"]), row["report_date"])
        incumbent = folded.get(key)
        rank = (
            row["public_availability_date"],
            row.get("filing_date") or dt.date.min,
            int(row.get("is_amendment") is True),
            str(row["filing_id"]),
        )
        if incumbent is None:
            folded[key] = row
            continue
        prior_rank = (
            incumbent["public_availability_date"],
            incumbent.get("filing_date") or dt.date.min,
            int(incumbent.get("is_amendment") is True),
            str(incumbent["filing_id"]),
        )
        if rank > prior_rank:
            folded[key] = row

    by_manager: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for (manager, _), row in folded.items():
        by_manager[manager].append(row)
    current_prior: dict[
        str, tuple[Mapping[str, Any], Mapping[str, Any] | None]
    ] = {}
    for manager, rows in by_manager.items():
        ordered = sorted(rows, key=lambda row: row["report_date"])
        current_candidates = [
            row
            for row in ordered
            if isinstance(row.get("stale_after_date"), dt.date)
            and signal_date <= row["stale_after_date"]
        ]
        if not current_candidates:
            continue
        current = current_candidates[-1]
        prior = next(
            (
                row
                for row in reversed(ordered)
                if row["report_date"] < current["report_date"]
            ),
            None,
        )
        current_prior[manager] = (current, prior)
    return all_filings, current_prior


def compute_13f_features(
    *,
    target_permno: int,
    signal_date: dt.date,
    filing_rows: Iterable[Mapping[str, Any]],
    holding_rows: Iterable[Mapping[str, Any]],
    identity_intervals: Iterable[Mapping[str, Any]],
    shares_intervals: Iterable[Mapping[str, Any]],
) -> dict[str, float | None]:
    names = (
        "institutional_ownership_ratio",
        "institutional_ownership_change_1q",
        "institutional_holder_breadth",
        "institutional_ownership_hhi",
    )
    all_filings, current_prior = _select_13f_snapshots(filing_rows, signal_date)
    if not all_filings or signal_date < min(row["public_availability_date"] for row in all_filings):
        return {name: None for name in names}

    if not current_prior:
        return {name: None for name in names}

    holdings_by_filing: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in holding_rows:
        holdings_by_filing[str(row["filing_id"])].append(row)
    identities = list(identity_intervals)
    share_rows = list(shares_intervals)
    current_positions: dict[str, tuple[float, float | None]] = {}
    change_terms: list[float] = []
    comparable_security_managers = 0
    change_invalid = False
    for manager, (current, prior) in current_prior.items():
        current_position = _target_position(
            current,
            target_permno=target_permno,
            holdings_by_filing=holdings_by_filing,
            identity_intervals=identities,
            shares_intervals=share_rows,
        )
        if current_position is None:
            return {name: None for name in names}
        current_positions[manager] = current_position
        if prior is None:
            continue
        prior_position = _target_position(
            prior,
            target_permno=target_permno,
            holdings_by_filing=holdings_by_filing,
            identity_intervals=identities,
            shares_intervals=share_rows,
        )
        if prior_position is None:
            change_invalid = True
            continue
        if current_position[0] > 0.0 or prior_position[0] > 0.0:
            comparable_security_managers += 1
            if current_position[1] is None or prior_position[1] is None:
                change_invalid = True
            else:
                change_terms.append(current_position[1] - prior_position[1])

    positive_shares = [position[0] for position in current_positions.values() if position[0] > 0.0]
    current_fractions = [position[1] for position in current_positions.values()]
    ratio = (
        None
        if any(fraction is None for fraction in current_fractions)
        else sum(float(fraction) for fraction in current_fractions)
    )
    total_shares = sum(positive_shares)
    hhi = None if total_shares <= 0.0 else sum(value * value for value in positive_shares) / (total_shares * total_shares)
    change = None if change_invalid or comparable_security_managers == 0 else sum(change_terms)
    return {
        "institutional_ownership_ratio": ratio,
        "institutional_ownership_change_1q": change,
        "institutional_holder_breadth": math.log1p(len(positive_shares)),
        "institutional_ownership_hhi": hhi,
    }


def compute_ownership_event_features(
    *,
    institutional: Mapping[str, float | None],
    sec_events: Mapping[str, float | int | str | list[str] | None],
    restatements: Mapping[str, int | None],
    identity: Mapping[str, float | int | str | None],
) -> dict[str, float | int | str | list[str] | None]:
    """Assemble the exact 32-feature family, preserving registry order."""

    values: dict[str, float | int | str | list[str] | None] = {}
    values.update(institutional)
    values.update(sec_events)
    values.update(restatements)
    values.update(
        {
            "business_segment_count": None,
            "geographic_segment_count": None,
            "business_segment_revenue_hhi": None,
            "business_segment_revenue_hhi_change_1y": None,
        }
    )
    values.update(identity)
    missing = set(FEATURE_NAMES) - set(values)
    extra = set(values) - set(FEATURE_NAMES)
    if missing or extra:
        raise OwnershipEventFeatureError(f"Ownership family mismatch: missing={sorted(missing)}, extra={sorted(extra)}")
    return {name: values[name] for name in FEATURE_NAMES}


def _max_row_date(
    rows: Iterable[Mapping[str, Any]], field: str
) -> dt.date | None:
    dates = [row.get(field) for row in rows]
    valid = [value for value in dates if isinstance(value, dt.date)]
    return max(valid) if valid else None


def _minimum_stale(rows: Iterable[Mapping[str, Any]]) -> dt.date | None:
    dates = [row.get("stale_after_date") for row in rows]
    valid = [value for value in dates if isinstance(value, dt.date)]
    return min(valid) if valid else None


def _link_missing_status(status: str) -> str:
    return (
        "MISSING_AMBIGUOUS_LINK"
        if status == "AMBIGUOUS"
        else "MISSING_NOT_COVERED"
    )


def build_ownership_event_feature_result(
    *,
    target_permno: int,
    signal_date: dt.date,
    filing_13f_rows: Iterable[Mapping[str, Any]],
    holding_13f_rows: Iterable[Mapping[str, Any]],
    identity_intervals: Iterable[Mapping[str, Any]],
    shares_intervals: Iterable[Mapping[str, Any]],
    sec_match_status: str,
    cik: str | None,
    sec_filing_rows: Iterable[Mapping[str, Any]],
    sec_item_rows: Iterable[Mapping[str, Any]],
    restatement_positively_covered: bool,
    restatement_event_rows: Iterable[Mapping[str, Any]],
    identity_row: Mapping[str, Any] | None,
    fundamental_fic: str | None,
    fundamental_loc: str | None,
    reporting_currency: str | None,
    fundamental_observation_date: dt.date | None,
    fundamental_public_availability_date: dt.date | None,
    fundamental_stale_after_date: dt.date | None,
    ff12_mapping: Mapping[str, Any],
) -> FeatureFamilyResult:
    """Build all 32 values with publication-ready per-feature lineage."""

    filings_13f = list(filing_13f_rows)
    holdings_13f = list(holding_13f_rows)
    identities = list(identity_intervals)
    shares = list(shares_intervals)
    sec_filings = list(sec_filing_rows)
    sec_items = list(sec_item_rows)
    restatement_events = list(restatement_event_rows)

    institutional = compute_13f_features(
        target_permno=target_permno,
        signal_date=signal_date,
        filing_rows=filings_13f,
        holding_rows=holdings_13f,
        identity_intervals=identities,
        shares_intervals=shares,
    )
    sec_events = compute_sec_event_features(
        sec_match_status=sec_match_status,
        cik=cik,
        signal_date=signal_date,
        filing_rows=sec_filings,
        item_rows=sec_items,
    )
    restatements = compute_restatement_features(
        positively_covered=restatement_positively_covered,
        signal_date=signal_date,
        event_rows=restatement_events,
    )
    identity = compute_identity_features(
        identity_row,
        fundamental_fic=fundamental_fic,
        fundamental_loc=fundamental_loc,
        reporting_currency=reporting_currency,
        ff12_mapping=ff12_mapping,
    )
    values = compute_ownership_event_features(
        institutional=institutional,
        sec_events=sec_events,
        restatements=restatements,
        identity=identity,
    )
    lineage: dict[str, FeatureLineage] = {}

    all_13f, snapshots = _select_13f_snapshots(filings_13f, signal_date)
    current_rows = [current for current, _ in snapshots.values()]
    comparison_rows = [
        row
        for current, prior in snapshots.values()
        for row in (current, prior)
        if row is not None
    ]
    if (
        not all_13f
        or signal_date
        < min(row["public_availability_date"] for row in all_13f)
    ):
        institutional_missing = "MISSING_SOURCE_UNAVAILABLE"
    elif not current_rows:
        institutional_missing = "MISSING_STALE"
    else:
        institutional_missing = "MISSING_INVALID_INPUT"
    for name in FEATURE_NAMES[:4]:
        contributors = (
            comparison_rows
            if name == "institutional_ownership_change_1q"
            else current_rows
        )
        missing_status = (
            "MISSING_INSUFFICIENT_HISTORY"
            if name == "institutional_ownership_change_1q"
            and current_rows
            and values[name] is None
            else institutional_missing
        )
        lineage[name] = make_lineage(
            values[name],
            _max_row_date(contributors, "observation_date"),
            _max_row_date(contributors, "public_availability_date"),
            _minimum_stale(current_rows),
            missing_status,
        )

    normalized_cik = normalize_text(cik)
    eligible_sec = [
        row
        for row in sec_filings
        if normalize_text(row.get("cik")) == normalized_cik
        and _eligible(row, signal_date)
    ]
    annual = _latest_filing(eligible_sec, ANNUAL_FORMS)
    interim = _latest_filing(eligible_sec, INTERIM_FORMS)
    link_missing = _link_missing_status(sec_match_status)
    for names, row in (
        (
            ("days_since_latest_annual_filing", "latest_annual_filing_form"),
            annual,
        ),
        (
            (
                "days_since_latest_interim_financial_filing",
                "latest_interim_financial_filing_form",
            ),
            interim,
        ),
    ):
        for name in names:
            unknown = isinstance(values[name], str) and values[name] == "UNKNOWN"
            lineage[name] = make_lineage(
                values[name],
                None if row is None else row.get("observation_date"),
                None if row is None else row.get("public_availability_date"),
                missing_status=(
                    link_missing
                    if sec_match_status != "MATCHED"
                    else "MISSING_INSUFFICIENT_HISTORY"
                ),
                is_missing=values[name] is None or unknown,
            )

    for form, stem in (("8-K", "8k"), ("6-K", "6k")):
        form_rows = [
            row
            for row in eligible_sec
            if normalize_text(row.get("form_family")) == form
        ]
        for days in (30, 90):
            window = _window_rows(form_rows, signal_date, days)
            for prefix in ("count", "recency"):
                name = f"{prefix}_{stem}_{days}d"
                has_value = values[name] is not None
                observation = _max_row_date(window, "observation_date")
                public = _max_row_date(window, "public_availability_date")
                if prefix == "count" and has_value and not window:
                    observation = public = signal_date
                lineage[name] = make_lineage(
                    values[name],
                    observation,
                    public,
                    missing_status=(
                        link_missing
                        if sec_match_status != "MATCHED"
                        else "MISSING_INSUFFICIENT_HISTORY"
                    ),
                )

    for name in (
        "restatement_disclosed_90d",
        "restatement_disclosed_365d",
    ):
        lineage[name] = make_lineage(
            values[name],
            signal_date,
            signal_date,
            missing_status="MISSING_NOT_COVERED",
        )
    for name in FEATURE_NAMES[18:22]:
        lineage[name] = make_lineage(
            values[name],
            None,
            None,
            missing_status="MISSING_SOURCE_UNAVAILABLE",
        )

    eligible_items = [
        row
        for row in sec_items
        if normalize_text(row.get("cik")) == normalized_cik
        and normalize_text(row.get("form_family")) == "8-K"
        and _eligible(row, signal_date)
    ]
    for days in (30, 90):
        name = f"eight_k_item_families_{days}d"
        window = _window_rows(eligible_items, signal_date, days)
        is_unknown = values[name] == ["UNKNOWN"]
        observation = _max_row_date(window, "observation_date")
        public = _max_row_date(window, "public_availability_date")
        if values[name] == ["NO_EVENT"]:
            observation = public = signal_date
        lineage[name] = make_lineage(
            values[name],
            observation,
            public,
            missing_status=link_missing,
            is_missing=is_unknown,
        )

    identity_names = FEATURE_NAMES[24:]
    fundamental_country_used = (
        _known_country(fundamental_fic) is not None
        or _known_country(fundamental_loc) is not None
    )
    for name in identity_names:
        value = values[name]
        unknown = (
            value is None
            or (isinstance(value, str) and value.startswith("UNKNOWN"))
        )
        if name == "reporting_currency_category":
            observation = fundamental_observation_date
            public = fundamental_public_availability_date
            stale = fundamental_stale_after_date
        elif name in {"domicile_country_category", "foreign_listing_indicator"} and fundamental_country_used:
            observation = fundamental_observation_date
            public = fundamental_public_availability_date
            stale = fundamental_stale_after_date
        else:
            observation = public = signal_date if identity_row is not None else None
            stale = None
        lineage[name] = make_lineage(
            value,
            observation,
            public,
            stale,
            "MISSING_NOT_COVERED" if identity_row is None else "MISSING_INVALID_INPUT",
            is_missing=unknown,
        )

    return FeatureFamilyResult(
        values=values,
        lineage={name: lineage[name] for name in FEATURE_NAMES},
    )
