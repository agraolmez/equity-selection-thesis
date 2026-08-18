"""Availability-lagged Compustat bundle selection and Stage 6 features."""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from portfolio_bot.features.feature_math import safe_ratio, signed_growth
from portfolio_bot.features.feature_lineage import (
    FeatureFamilyResult,
    FeatureLineage,
    make_lineage,
)


MONETARY_SCALE = 1_000_000.0
QUARTERLY_DISCRETE = {
    "income": "ibq",
    "sales": "saleq",
    "cogs": "cogsq",
    "ebitda": "oibdpq",
    "operating_income_after_depreciation": "oiadpq",
    "research_and_development": "xrdq",
}
QUARTERLY_YTD = {
    "operating_cash_flow": "oancfy",
    "capital_expenditure": "capxy",
    "dividends": "dvy",
    "stock_issuance": "sstky",
    "stock_repurchases": "prstkcy",
}
QUARTERLY_BALANCE = {
    "assets": "atq",
    "common_equity": "ceqq",
    "cash": "cheq",
    "short_debt": "dlcq",
    "long_debt": "dlttq",
    "current_assets": "actq",
    "current_liabilities": "lctq",
    "net_property_plant_equipment": "ppentq",
}
ANNUAL_FIELDS: dict[str, tuple[str, ...]] = {
    "income": ("ib",),
    "sales": ("sale",),
    "cogs": ("cogs",),
    "ebitda": ("ebitda", "oibdp"),
    "operating_income_after_depreciation": ("oiadp",),
    "research_and_development": ("xrd",),
    "operating_cash_flow": ("oancf",),
    "capital_expenditure": ("capx",),
    "dividends": ("dvc",),
    "stock_issuance": ("sstk",),
    "stock_repurchases": ("prstkc",),
    "assets": ("at",),
    "common_equity": ("ceq",),
    "cash": ("che",),
    "short_debt": ("dlc",),
    "long_debt": ("dltt",),
    "current_assets": ("act",),
    "current_liabilities": ("lct",),
    "net_property_plant_equipment": ("ppent",),
    "shares_outstanding": ("csho",),
}

FEATURE_NAMES = (
    "earnings_to_price",
    "positive_earnings_pe",
    "book_to_market",
    "sales_to_price",
    "operating_cash_flow_to_price",
    "ebitda_to_enterprise_value",
    "dividend_yield",
    "return_on_assets",
    "return_on_equity",
    "gross_profitability",
    "operating_margin",
    "asset_turnover",
    "accruals_to_average_assets",
    "capital_expenditure_to_assets",
    "research_and_development_to_sales",
    "investment_growth",
    "total_debt_to_assets",
    "long_term_debt_to_assets",
    "current_ratio",
    "cash_to_assets",
    "net_equity_issuance",
    "repurchase_payout_yield",
    "sales_growth_yoy",
    "earnings_growth_yoy",
    "asset_growth_yoy",
    "share_growth_yoy",
    "loss_indicator",
    "accounting_market_currency_verified",
    "fundamental_source_basis",
)


class FundamentalFeatureError(RuntimeError):
    """Raised when the canonical Compustat fiscal grain is ambiguous."""


@dataclass(frozen=True)
class FundamentalBundle:
    basis: str
    current_rows: tuple[Mapping[str, Any], ...]
    prior_rows: tuple[Mapping[str, Any], ...]
    all_quarters: Mapping[tuple[int, int], Mapping[str, Any]]

    @property
    def anchor(self) -> Mapping[str, Any]:
        return self.current_rows[-1]


@dataclass(frozen=True)
class FundamentalResult(FeatureFamilyResult):
    observation_date: dt.date | None
    public_availability_date: dt.date | None
    stale_after_date: dt.date | None
    currency_code: str | None
    fic: str | None
    loc: str | None


def _number(value: Any) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _normalized_text(value: Any) -> str | None:
    if value is None:
        return None
    result = str(value).strip().upper()
    return result or None


def _first_number(row: Mapping[str, Any], columns: Iterable[str]) -> float | None:
    for column in columns:
        value = _number(row.get(column))
        if value is not None:
            return value
    return None


def validate_unique_fiscal_rows(
    quarterly_rows: Iterable[Mapping[str, Any]],
    annual_rows: Iterable[Mapping[str, Any]],
) -> None:
    quarterly_seen: set[tuple[str, int, int, int]] = set()
    for row in quarterly_rows:
        if row.get("standard_format_eligible") is not True:
            continue
        key = (
            str(row["gvkey"]),
            int(row["permno"]),
            int(row["fyearq"]),
            int(row["fqtr"]),
        )
        if key in quarterly_seen:
            raise FundamentalFeatureError(f"Duplicate quarterly fiscal key: {key}")
        quarterly_seen.add(key)
    annual_seen: set[tuple[str, int, int]] = set()
    for row in annual_rows:
        if row.get("standard_format_eligible") is not True:
            continue
        key = (str(row["gvkey"]), int(row["permno"]), int(row["fyear"]))
        if key in annual_seen:
            raise FundamentalFeatureError(f"Duplicate annual fiscal key: {key}")
        annual_seen.add(key)


def _known_by_signal(row: Mapping[str, Any], signal_date: dt.date) -> bool:
    observation = row.get("observation_date")
    availability = row.get("public_availability_date")
    return (
        row.get("standard_format_eligible") is True
        and isinstance(observation, dt.date)
        and isinstance(availability, dt.date)
        and observation <= signal_date
        and availability <= signal_date
    )


def select_fundamental_bundle(
    quarterly_rows: Iterable[Mapping[str, Any]],
    annual_rows: Iterable[Mapping[str, Any]],
    signal_date: dt.date,
) -> FundamentalBundle | None:
    quarterly = [row for row in quarterly_rows if _known_by_signal(row, signal_date)]
    annual = [row for row in annual_rows if _known_by_signal(row, signal_date)]
    validate_unique_fiscal_rows(quarterly, annual)

    quarter_index = {
        (int(row["fyearq"]), int(row["fqtr"])): row for row in quarterly
    }
    candidates = sorted(
        quarterly,
        key=lambda row: (
            int(row["fyearq"]) * 4 + int(row["fqtr"]) - 1,
            row["datadate"],
        ),
        reverse=True,
    )
    for anchor in candidates:
        stale_after = anchor.get("stale_after_date")
        if not isinstance(stale_after, dt.date) or signal_date > stale_after:
            continue
        anchor_ordinal = int(anchor["fyearq"]) * 4 + int(anchor["fqtr"]) - 1
        current: list[Mapping[str, Any]] = []
        for ordinal in range(anchor_ordinal - 3, anchor_ordinal + 1):
            key = (ordinal // 4, ordinal % 4 + 1)
            row = quarter_index.get(key)
            if row is None:
                current = []
                break
            current.append(row)
        if len(current) != 4 or len({row.get("fyr") for row in current}) != 1:
            continue
        prior: list[Mapping[str, Any]] = []
        for ordinal in range(anchor_ordinal - 7, anchor_ordinal - 3):
            key = (ordinal // 4, ordinal % 4 + 1)
            row = quarter_index.get(key)
            if row is None:
                prior = []
                break
            prior.append(row)
        return FundamentalBundle(
            basis="TTM",
            current_rows=tuple(current),
            prior_rows=tuple(prior),
            all_quarters=quarter_index,
        )

    annual_candidates = sorted(
        annual,
        key=lambda row: (int(row["fyear"]), row["datadate"]),
        reverse=True,
    )
    for anchor in annual_candidates:
        stale_after = anchor.get("stale_after_date")
        if not isinstance(stale_after, dt.date) or signal_date > stale_after:
            continue
        prior = next(
            (
                row
                for row in annual_candidates
                if int(row["fyear"]) == int(anchor["fyear"]) - 1
            ),
            None,
        )
        return FundamentalBundle(
            basis="ANNUAL_FALLBACK",
            current_rows=(anchor,),
            prior_rows=() if prior is None else (prior,),
            all_quarters={},
        )
    return None


def _quarter_ytd_discrete(
    row: Mapping[str, Any],
    column: str,
    quarter_index: Mapping[tuple[int, int], Mapping[str, Any]],
) -> float | None:
    value = _number(row.get(column))
    if value is None:
        return None
    quarter = int(row["fqtr"])
    if quarter == 1:
        return value
    prior = quarter_index.get((int(row["fyearq"]), quarter - 1))
    if prior is None:
        return None
    prior_value = _number(prior.get(column))
    return None if prior_value is None else value - prior_value


def _quarter_values(
    rows: tuple[Mapping[str, Any], ...],
    quarter_index: Mapping[tuple[int, int], Mapping[str, Any]],
) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for concept, column in QUARTERLY_DISCRETE.items():
        values = [_number(row.get(column)) for row in rows]
        result[concept] = None if any(value is None for value in values) else float(sum(values))
    for concept, column in QUARTERLY_YTD.items():
        values = [_quarter_ytd_discrete(row, column, quarter_index) for row in rows]
        result[concept] = None if any(value is None for value in values) else float(sum(values))
    endpoint = rows[-1]
    for concept, column in QUARTERLY_BALANCE.items():
        result[concept] = _number(endpoint.get(column))
    result["shares_outstanding"] = None
    return result


def _annual_values(row: Mapping[str, Any]) -> dict[str, float | None]:
    return {concept: _first_number(row, columns) for concept, columns in ANNUAL_FIELDS.items()}


def _average(current: float | None, prior: float | None) -> float | None:
    if current is None or prior is None:
        return None
    result = (current + prior) / 2.0
    return result if math.isfinite(result) else None


def _scaled(value: float | None) -> float | None:
    return None if value is None else value * MONETARY_SCALE


def _unique_rows(
    rows: Iterable[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    result: list[Mapping[str, Any]] = []
    seen: set[int] = set()
    for row in rows:
        marker = id(row)
        if marker not in seen:
            seen.add(marker)
            result.append(row)
    return tuple(result)


def _quarter_contributor_rows(
    rows: tuple[Mapping[str, Any], ...],
    quarter_index: Mapping[tuple[int, int], Mapping[str, Any]],
) -> dict[str, tuple[Mapping[str, Any], ...]]:
    result = {
        concept: tuple(rows) for concept in QUARTERLY_DISCRETE
    }
    for concept in QUARTERLY_YTD:
        contributors: list[Mapping[str, Any]] = []
        for row in rows:
            contributors.append(row)
            quarter = int(row["fqtr"])
            if quarter > 1:
                prior = quarter_index.get((int(row["fyearq"]), quarter - 1))
                if prior is not None:
                    contributors.append(prior)
        result[concept] = _unique_rows(contributors)
    endpoint = rows[-1]
    result.update({concept: (endpoint,) for concept in QUARTERLY_BALANCE})
    result["shares_outstanding"] = ()
    return result


def _annual_contributor_rows(
    row: Mapping[str, Any],
) -> dict[str, tuple[Mapping[str, Any], ...]]:
    return {concept: (row,) for concept in ANNUAL_FIELDS}


def _lineage_for_value(
    value: object,
    rows: Iterable[Mapping[str, Any]],
    *,
    stale_after_date: dt.date | None,
    missing_status: str,
    extra_observation_date: dt.date | None = None,
    extra_public_availability_date: dt.date | None = None,
    is_missing: bool | None = None,
) -> FeatureLineage:
    contributors = _unique_rows(rows)
    observation_dates = [
        row["observation_date"]
        for row in contributors
        if isinstance(row.get("observation_date"), dt.date)
    ]
    availability_dates = [
        row["public_availability_date"]
        for row in contributors
        if isinstance(row.get("public_availability_date"), dt.date)
    ]
    if isinstance(extra_observation_date, dt.date):
        observation_dates.append(extra_observation_date)
    if isinstance(extra_public_availability_date, dt.date):
        availability_dates.append(extra_public_availability_date)
    return make_lineage(
        value,
        max(observation_dates) if observation_dates else None,
        max(availability_dates) if availability_dates else None,
        stale_after_date,
        missing_status,
        is_missing=is_missing,
    )


def compute_fundamental_features(
    bundle: FundamentalBundle | None,
    market_cap_usd: float | None,
    *,
    market_observation_date: dt.date | None = None,
    market_public_availability_date: dt.date | None = None,
    bundle_missing_status: str = "MISSING_SOURCE_UNAVAILABLE",
) -> FundamentalResult:
    values: dict[str, float | int | str | None] = {name: None for name in FEATURE_NAMES}
    if bundle is None:
        values["fundamental_source_basis"] = "UNKNOWN"
        lineage = {
            name: _lineage_for_value(
                value,
                (),
                stale_after_date=None,
                missing_status=bundle_missing_status,
                is_missing=True,
            )
            for name, value in values.items()
        }
        return FundamentalResult(
            values,
            lineage,
            None,
            None,
            None,
            None,
            None,
            None,
        )

    anchor = bundle.anchor
    if bundle.basis == "TTM":
        current = _quarter_values(bundle.current_rows, bundle.all_quarters)
        current_contributors = _quarter_contributor_rows(
            bundle.current_rows, bundle.all_quarters
        )
        prior = (
            _quarter_values(bundle.prior_rows, bundle.all_quarters)
            if len(bundle.prior_rows) == 4
            else {key: None for key in current}
        )
        prior_contributors = (
            _quarter_contributor_rows(bundle.prior_rows, bundle.all_quarters)
            if len(bundle.prior_rows) == 4
            else {key: () for key in current}
        )
        beginning = bundle.all_quarters.get(
            (int(anchor["fyearq"]) - 1, int(anchor["fqtr"]))
        )
        beginning_values = (
            {concept: _number(beginning.get(column)) for concept, column in QUARTERLY_BALANCE.items()}
            if beginning is not None
            else {concept: None for concept in QUARTERLY_BALANCE}
        )
        beginning_contributors = {
            concept: (() if beginning is None else (beginning,))
            for concept in QUARTERLY_BALANCE
        }
    else:
        current = _annual_values(bundle.current_rows[0])
        current_contributors = _annual_contributor_rows(bundle.current_rows[0])
        prior = _annual_values(bundle.prior_rows[0]) if bundle.prior_rows else {key: None for key in current}
        prior_contributors = (
            _annual_contributor_rows(bundle.prior_rows[0])
            if bundle.prior_rows
            else {key: () for key in current}
        )
        beginning_values = prior
        beginning_contributors = prior_contributors

    currency = _normalized_text(anchor.get("currency_code"))
    if bundle.basis == "TTM":
        currencies = {_normalized_text(row.get("currency_code")) for row in bundle.current_rows}
        currency = next(iter(currencies)) if len(currencies) == 1 and None not in currencies else None
    accounting_currency_valid = currency is not None
    market_cap = _number(market_cap_usd)
    market_dates_valid = all(
        isinstance(value, dt.date) and not isinstance(value, dt.datetime)
        for value in (
            market_observation_date,
            market_public_availability_date,
        )
    )
    usd_market = (
        market_cap
        if currency == "USD"
        and market_cap is not None
        and market_cap > 0.0
        and market_dates_valid
        else None
    )

    average_assets = _average(current["assets"], beginning_values.get("assets"))
    average_equity = _average(current["common_equity"], beginning_values.get("common_equity"))
    income = current["income"]
    sales = current["sales"]
    operating_cash_flow = current["operating_cash_flow"]
    ebitda = current["ebitda"]
    enterprise_value = None
    if usd_market is not None and all(
        current[name] is not None for name in ("short_debt", "long_debt", "cash")
    ):
        enterprise_value = usd_market + _scaled(current["short_debt"]) + _scaled(current["long_debt"]) - _scaled(current["cash"])

    values.update(
        {
            "earnings_to_price": safe_ratio(_scaled(income), usd_market),
            "positive_earnings_pe": safe_ratio(usd_market, _scaled(income)) if income is not None and income > 0.0 else None,
            "book_to_market": safe_ratio(_scaled(current["common_equity"]), usd_market),
            "sales_to_price": safe_ratio(_scaled(sales), usd_market),
            "operating_cash_flow_to_price": safe_ratio(_scaled(operating_cash_flow), usd_market),
            "ebitda_to_enterprise_value": safe_ratio(_scaled(ebitda), enterprise_value),
            "dividend_yield": safe_ratio(_scaled(current["dividends"]), usd_market),
            "return_on_assets": safe_ratio(income, average_assets),
            "return_on_equity": safe_ratio(income, average_equity),
            "gross_profitability": safe_ratio(
                None if sales is None or current["cogs"] is None else sales - current["cogs"],
                average_assets,
            ),
            "operating_margin": safe_ratio(current["operating_income_after_depreciation"], sales),
            "asset_turnover": safe_ratio(sales, average_assets),
            "accruals_to_average_assets": safe_ratio(
                None if income is None or operating_cash_flow is None else income - operating_cash_flow,
                average_assets,
            ),
            "capital_expenditure_to_assets": safe_ratio(current["capital_expenditure"], average_assets),
            "research_and_development_to_sales": safe_ratio(current["research_and_development"], sales),
            "investment_growth": safe_ratio(
                None
                if current["net_property_plant_equipment"] is None or prior["net_property_plant_equipment"] is None
                else current["net_property_plant_equipment"] - prior["net_property_plant_equipment"],
                prior["assets"],
            ),
            "total_debt_to_assets": safe_ratio(
                None
                if current["short_debt"] is None or current["long_debt"] is None
                else current["short_debt"] + current["long_debt"],
                current["assets"],
            ),
            "long_term_debt_to_assets": safe_ratio(current["long_debt"], current["assets"]),
            "current_ratio": safe_ratio(current["current_assets"], current["current_liabilities"]),
            "cash_to_assets": safe_ratio(current["cash"], current["assets"]),
            "net_equity_issuance": safe_ratio(
                None
                if current["stock_issuance"] is None or current["stock_repurchases"] is None
                else current["stock_issuance"] - current["stock_repurchases"],
                average_assets,
            ),
            "repurchase_payout_yield": safe_ratio(
                None
                if current["stock_repurchases"] is None or current["dividends"] is None
                else _scaled(current["stock_repurchases"] + current["dividends"]),
                usd_market,
            ),
            "sales_growth_yoy": signed_growth(sales, prior["sales"], positive=True),
            "earnings_growth_yoy": signed_growth(income, prior["income"], positive=False),
            "asset_growth_yoy": signed_growth(current["assets"], prior["assets"], positive=True),
            "share_growth_yoy": signed_growth(current["shares_outstanding"], prior["shares_outstanding"], positive=True),
            "loss_indicator": None if income is None else int(income <= 0.0),
            "accounting_market_currency_verified": None if currency is None else int(currency == "USD"),
            "fundamental_source_basis": bundle.basis,
        }
    )
    if not accounting_currency_valid:
        for name in FEATURE_NAMES:
            if name not in {
                "accounting_market_currency_verified",
                "fundamental_source_basis",
            }:
                values[name] = None

    def _concept_rows(
        source: Mapping[str, tuple[Mapping[str, Any], ...]],
        *concepts: str,
    ) -> tuple[Mapping[str, Any], ...]:
        return _unique_rows(
            row for concept in concepts for row in source.get(concept, ())
        )

    average_asset_rows = _unique_rows(
        (*_concept_rows(current_contributors, "assets"),
         *_concept_rows(beginning_contributors, "assets"))
    )
    average_equity_rows = _unique_rows(
        (*_concept_rows(current_contributors, "common_equity"),
         *_concept_rows(beginning_contributors, "common_equity"))
    )
    dependencies: dict[str, tuple[Mapping[str, Any], ...]] = {
        "earnings_to_price": _concept_rows(current_contributors, "income"),
        "positive_earnings_pe": _concept_rows(current_contributors, "income"),
        "book_to_market": _concept_rows(current_contributors, "common_equity"),
        "sales_to_price": _concept_rows(current_contributors, "sales"),
        "operating_cash_flow_to_price": _concept_rows(
            current_contributors, "operating_cash_flow"
        ),
        "ebitda_to_enterprise_value": _concept_rows(
            current_contributors,
            "ebitda",
            "short_debt",
            "long_debt",
            "cash",
        ),
        "dividend_yield": _concept_rows(current_contributors, "dividends"),
        "return_on_assets": _unique_rows(
            (*_concept_rows(current_contributors, "income"), *average_asset_rows)
        ),
        "return_on_equity": _unique_rows(
            (*_concept_rows(current_contributors, "income"), *average_equity_rows)
        ),
        "gross_profitability": _unique_rows(
            (*_concept_rows(current_contributors, "sales", "cogs"), *average_asset_rows)
        ),
        "operating_margin": _concept_rows(
            current_contributors, "operating_income_after_depreciation", "sales"
        ),
        "asset_turnover": _unique_rows(
            (*_concept_rows(current_contributors, "sales"), *average_asset_rows)
        ),
        "accruals_to_average_assets": _unique_rows(
            (
                *_concept_rows(current_contributors, "income", "operating_cash_flow"),
                *average_asset_rows,
            )
        ),
        "capital_expenditure_to_assets": _unique_rows(
            (*_concept_rows(current_contributors, "capital_expenditure"), *average_asset_rows)
        ),
        "research_and_development_to_sales": _concept_rows(
            current_contributors, "research_and_development", "sales"
        ),
        "investment_growth": _unique_rows(
            (
                *_concept_rows(current_contributors, "net_property_plant_equipment"),
                *_concept_rows(
                    prior_contributors, "net_property_plant_equipment", "assets"
                ),
            )
        ),
        "total_debt_to_assets": _concept_rows(
            current_contributors, "short_debt", "long_debt", "assets"
        ),
        "long_term_debt_to_assets": _concept_rows(
            current_contributors, "long_debt", "assets"
        ),
        "current_ratio": _concept_rows(
            current_contributors, "current_assets", "current_liabilities"
        ),
        "cash_to_assets": _concept_rows(current_contributors, "cash", "assets"),
        "net_equity_issuance": _unique_rows(
            (
                *_concept_rows(current_contributors, "stock_issuance", "stock_repurchases"),
                *average_asset_rows,
            )
        ),
        "repurchase_payout_yield": _concept_rows(
            current_contributors, "stock_repurchases", "dividends"
        ),
        "sales_growth_yoy": _unique_rows(
            (
                *_concept_rows(current_contributors, "sales"),
                *_concept_rows(prior_contributors, "sales"),
            )
        ),
        "earnings_growth_yoy": _unique_rows(
            (
                *_concept_rows(current_contributors, "income"),
                *_concept_rows(prior_contributors, "income"),
            )
        ),
        "asset_growth_yoy": _unique_rows(
            (
                *_concept_rows(current_contributors, "assets"),
                *_concept_rows(prior_contributors, "assets"),
            )
        ),
        "share_growth_yoy": _unique_rows(
            (
                *_concept_rows(current_contributors, "shares_outstanding"),
                *_concept_rows(prior_contributors, "shares_outstanding"),
            )
        ),
        "loss_indicator": _concept_rows(current_contributors, "income"),
        "accounting_market_currency_verified": tuple(bundle.current_rows),
        "fundamental_source_basis": tuple(bundle.current_rows),
    }
    market_features = {
        "earnings_to_price",
        "positive_earnings_pe",
        "book_to_market",
        "sales_to_price",
        "operating_cash_flow_to_price",
        "ebitda_to_enterprise_value",
        "dividend_yield",
        "repurchase_payout_yield",
    }
    growth_features = {
        "investment_growth",
        "sales_growth_yoy",
        "earnings_growth_yoy",
        "asset_growth_yoy",
        "share_growth_yoy",
    }
    average_features = {
        "return_on_assets",
        "gross_profitability",
        "asset_turnover",
        "accruals_to_average_assets",
        "capital_expenditure_to_assets",
        "net_equity_issuance",
    }
    lineage: dict[str, FeatureLineage] = {}
    stale_after = anchor.get("stale_after_date")
    for name in FEATURE_NAMES:
        missing_status = "MISSING_INVALID_INPUT"
        if name == "share_growth_yoy" and bundle.basis == "TTM":
            missing_status = "MISSING_SOURCE_UNAVAILABLE"
        elif not accounting_currency_valid and name != "fundamental_source_basis":
            missing_status = "MISSING_INVALID_INPUT"
        elif name in growth_features and not bundle.prior_rows:
            missing_status = "MISSING_INSUFFICIENT_HISTORY"
        elif name in average_features and not _concept_rows(
            beginning_contributors, "assets"
        ):
            missing_status = "MISSING_INSUFFICIENT_HISTORY"
        elif name == "return_on_equity" and not _concept_rows(
            beginning_contributors, "common_equity"
        ):
            missing_status = "MISSING_INSUFFICIENT_HISTORY"
        lineage[name] = _lineage_for_value(
            values[name],
            dependencies[name],
            stale_after_date=stale_after if isinstance(stale_after, dt.date) else None,
            missing_status=missing_status,
            extra_observation_date=(
                market_observation_date if name in market_features else None
            ),
            extra_public_availability_date=(
                market_public_availability_date if name in market_features else None
            ),
        )
    return FundamentalResult(
        values=values,
        lineage=lineage,
        observation_date=max(row["observation_date"] for row in bundle.current_rows),
        public_availability_date=max(row["public_availability_date"] for row in bundle.current_rows),
        stale_after_date=anchor.get("stale_after_date"),
        currency_code=currency,
        fic=_normalized_text(anchor.get("fic")),
        loc=_normalized_text(anchor.get("loc")),
    )
