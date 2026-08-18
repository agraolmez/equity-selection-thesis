"""Fail-closed Stage 13 treatment for executable public-interest GDR exits.

The frozen CRSP reconciliation correctly leaves a post-delisting payoff
missing when CRSP has no later price or payment.  A very narrow subset still
has an authenticated, active, regular-way final exchange session.  This
module authenticates that complete evidence chain so Stage 13 may use the
ordinary final-session return and retire the position at that last tradable
close without relabelling the missing vendor delisting return as zero.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
from dataclasses import dataclass
from decimal import Decimal, localcontext
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import pyarrow as pa
import pyarrow.compute as pc


class TerminalEventOverlayError(RuntimeError):
    """Raised when the terminal-event policy or its evidence differs."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TerminalEventOverlayError(message)


def _jsonable(value: object) -> object:
    if isinstance(value, (dt.date, dt.datetime)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return (
            json.dumps(
                _jsonable(value),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise TerminalEventOverlayError(
            "terminal-event evidence is not finite canonical JSON"
        ) from exc


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _finite_decimal(value: object, *, label: str) -> Decimal:
    _require(isinstance(value, Decimal) and value.is_finite(), f"{label} differs")
    return value


def _date(value: object, *, label: str) -> dt.date:
    _require(type(value) is dt.date, f"{label} differs")
    return value


def _required_columns(table: pa.Table, names: Sequence[str], *, label: str) -> None:
    _require(isinstance(table, pa.Table), f"{label} is not an Arrow table")
    missing = sorted(set(names) - set(table.column_names))
    _require(not missing, f"{label} omits columns: {', '.join(missing)}")


def _unique_rows_by_key(
    table: pa.Table,
    *,
    key_names: tuple[str, ...],
    label: str,
) -> Mapping[tuple[object, ...], Mapping[str, object]]:
    rows: dict[tuple[object, ...], Mapping[str, object]] = {}
    for raw in table.to_pylist():
        row = MappingProxyType(dict(raw))
        key = tuple(row[name] for name in key_names)
        _require(key not in rows, f"{label} contains a duplicate key")
        rows[key] = row
    return MappingProxyType(rows)


@dataclass(frozen=True, slots=True)
class LastTradableCloseEvent:
    """One fully authenticated final-close liquidation overlay event."""

    permno: int
    delisting_date: dt.date
    posting_date: dt.date
    outcome_availability_date: dt.date
    ordinary_total_return: Decimal
    event_day_bid_usd_per_share: Decimal
    event_day_ask_usd_per_share: Decimal
    event_day_quoted_full_spread: Decimal
    source_event_sha256: str
    evidence_sha256: str

    def as_dict(self) -> dict[str, object]:
        return {
            "permno": self.permno,
            "delisting_date": self.delisting_date.isoformat(),
            "posting_date": self.posting_date.isoformat(),
            "outcome_availability_date": self.outcome_availability_date.isoformat(),
            "ordinary_total_return": format(self.ordinary_total_return, "f"),
            "event_day_bid_usd_per_share": format(
                self.event_day_bid_usd_per_share, "f"
            ),
            "event_day_ask_usd_per_share": format(
                self.event_day_ask_usd_per_share, "f"
            ),
            "event_day_quoted_full_spread": format(
                self.event_day_quoted_full_spread, "f"
            ),
            "source_event_sha256": self.source_event_sha256,
            "evidence_sha256": self.evidence_sha256,
        }


@dataclass(frozen=True, slots=True)
class LastTradableCloseOverlay:
    """Authenticated events and their content-bound audit receipt."""

    policy_id: str
    policy_sha256: str
    treatment_status: str
    events: tuple[LastTradableCloseEvent, ...]
    receipt: Mapping[str, object]

    @property
    def events_by_return_key(
        self,
    ) -> Mapping[tuple[int, dt.date], LastTradableCloseEvent]:
        return MappingProxyType(
            {(event.permno, event.delisting_date): event for event in self.events}
        )


_DELISTING_COLUMNS = (
    "permno",
    "delisting_date",
    "posting_date",
    "delisting_price",
    "delisting_price_flag",
    "action_type",
    "status_type",
    "reason_type",
    "payment_type",
    "successor_permno",
    "successor_permco",
    "observed_delisting_return",
    "return_missing_type",
    "next_date",
    "next_price",
    "next_price_flag",
    "amount_date",
    "delisting_dividend_amount",
    "distribution_type",
    "primary_exchange",
    "performance_related",
    "embedding_status",
    "treatment_status",
    "effective_delisting_return",
    "imputation_applied",
    "assigned_return_date",
    "outcome_availability_date",
    "assigned_exactly_once",
)

_RETURN_COLUMNS = (
    "permno",
    "calendar_date",
    "raw_total_return",
    "raw_return_ex_distributions",
    "raw_return_missing_flag",
    "raw_return_duration_flag",
    "mapped_trading_periods",
    "return_leg_class",
    "return_span_start_date",
    "return_span_end_date",
    "covered_trading_sessions",
    "span_alignment_status",
    "distribution_event_count",
    "delisting_event_count",
    "delisting_treatment_status",
    "delisting_adjustment_return",
    "authoritative_total_return",
    "outcome_availability_date",
    "authoritative_return_state",
    "reconciliation_status",
    "return_history_usable",
    "return_history_authorized",
)

_MARKET_COLUMNS = (
    "permno",
    "calendar_date",
    "conditional_type",
    "source_trading_status_flag",
    "price_flag",
    "close_usd_per_share",
    "bid_usd_per_share",
    "ask_usd_per_share",
    "volume_shares",
    "screening_price_valid",
    "liquidity_session_valid",
)

_MONTHLY_COLUMNS = (
    "permno",
    "calendar_month",
    "raw_monthly_total_return",
    "raw_monthly_return_flag",
    "monthly_flag_class",
    "compounded_raw_numeric_daily_total_return",
    "daily_observation_count",
    "usable_daily_return_count",
    "missing_daily_return_count",
    "delisting_event_count",
    "daily_coverage_class",
    "numeric_transport_absolute_difference",
    "tolerance_used",
    "numeric_transport_status",
    "authorized_return_diagnostic_status",
    "reconciliation_status",
)


def _structural_event_candidate(
    row: Mapping[str, object],
    *,
    policy: Mapping[str, object],
) -> bool:
    rule = policy["event_eligibility"]
    assert isinstance(rule, Mapping)
    price = row["delisting_price"]
    positive_price = (
        isinstance(price, Decimal) and price.is_finite() and price > Decimal(0)
    )
    return bool(
        row["action_type"] == rule["action_type"]
        and row["status_type"] == rule["status_type"]
        and row["reason_type"] == rule["reason_type"]
        and row["payment_type"] == rule["payment_type"]
        and row["distribution_type"] == rule["distribution_type"]
        and row["primary_exchange"] == rule["primary_exchange"]
        and row["performance_related"] is rule["performance_related"]
        and row["treatment_status"] == rule["treatment_status"]
        and row["embedding_status"] == rule["embedding_status"]
        and row["return_missing_type"] == rule["return_missing_type"]
        and row["observed_delisting_return"] is None
        and row["effective_delisting_return"] is None
        and row["successor_permno"] is None
        and row["successor_permco"] is None
        and row["next_date"] is None
        and row["next_price"] is None
        and row["next_price_flag"] == rule["next_price_flag"]
        and row["amount_date"] is None
        and row["delisting_dividend_amount"] is None
        and row["delisting_price_flag"] == rule["delisting_price_flag"]
        and positive_price
        and row["imputation_applied"] is False
        and row["assigned_return_date"] == row["delisting_date"]
        and row["outcome_availability_date"] == row["posting_date"]
        and row["assigned_exactly_once"] is True
    )


def _policy_identity(policy: Mapping[str, object], policy_sha256: str) -> tuple[
    str,
    Mapping[str, object],
    Mapping[str, object],
    Mapping[str, object],
    Mapping[str, object],
]:
    _require(policy.get("schema_version") == 1, "terminal-event policy schema differs")
    policy_id = policy.get("policy_id")
    _require(
        isinstance(policy_id, str)
        and policy_id == "stage13-public-interest-last-tradable-close-liquidation-v1",
        "terminal-event policy identity differs",
    )
    _require(
        isinstance(policy_sha256, str)
        and len(policy_sha256) == 64
        and all(character in "0123456789abcdef" for character in policy_sha256),
        "terminal-event policy SHA-256 differs",
    )
    scope = policy.get("scope")
    event_rule = policy.get("event_eligibility")
    session_rule = policy.get("final_session_eligibility")
    monthly_rule = policy.get("monthly_crosscheck")
    treatment = policy.get("treatment")
    sensitivity = policy.get("sensitivity")
    _require(
        all(
            isinstance(value, Mapping)
            for value in (
                scope,
                event_rule,
                session_rule,
                monthly_rule,
                treatment,
                sensitivity,
            )
        ),
        "terminal-event policy sections differ",
    )
    assert isinstance(scope, Mapping)
    assert isinstance(event_rule, Mapping)
    assert isinstance(session_rule, Mapping)
    assert isinstance(monthly_rule, Mapping)
    assert isinstance(treatment, Mapping)
    assert isinstance(sensitivity, Mapping)
    _require(
        scope.get("unexpected_or_missing_match_disposition") == "fail_closed",
        "terminal-event population is not fail-closed",
    )
    _require(
        treatment.get("ordinary_final_session_return_authorized") is True
        and treatment.get("vendor_post_delisting_return_relabelled_as_zero") is False
        and treatment.get("vendor_missing_return_fields_preserved_in_source_evidence")
        is True
        and treatment.get("settlement_kind") == "cash"
        and float(treatment.get("settlement_value_multiplier", math.nan)) == 1.0
        and treatment.get("terminal_total_return_already_applied") is True
        and treatment.get("apply_order")
        == "observed_final_session_return_then_single_cash_retirement",
        "terminal-event economic treatment differs",
    )
    _require(
        sensitivity.get("status")
        == "mandatory_reporting_only_one_at_a_time_counterfactuals"
        and sensitivity.get("counterfactual_post_delisting_value_multipliers")
        == [0.7, 0.45, 0.0]
        and sensitivity.get("counterfactual_multiplier_interpretation")
        == "replace_multiplier_one_final_close_cash_settlement_after_observed_ordinary_final_session_return"
        and sensitivity.get(
            "counterfactual_multiplier_applies_to_signed_post_return_pre_settlement_position_value"
        )
        is True
        and sensitivity.get(
            "counterfactual_multiplier_is_not_an_imputed_vendor_delisting_return"
        )
        is True
        and sensitivity.get("vendor_missing_return_fields_remain_missing") is True
        and sensitivity.get("terminal_execution_cost_stress")
        == "maximum_of_authenticated_event_day_quoted_full_spread_and_frozen_full_spread_20d"
        and sensitivity.get("event_day_quoted_full_spread_formula")
        == "(ask_usd_per_share-bid_usd_per_share)/((ask_usd_per_share+bid_usd_per_share)/2)"
        and sensitivity.get("execution_cost_charge")
        == "one_full_selected_spread_times_absolute_post_return_pre_settlement_position_value"
        and sensitivity.get(
            "execution_cost_scenario_is_separate_one_at_a_time_not_cartesian"
        )
        is True
        and sensitivity.get("within_holding_month_cash_carry")
        == "counterfactual_cash_difference_compounds_by_observed_remaining_daily_risk_free_growth_after_event_session"
        and sensitivity.get("subsequent_primary_monthly_percentage_returns_unchanged")
        is True
        and sensitivity.get("primary_estimates_modified") is False
        and sensitivity.get("model_fitting_scoring_selection_or_allocation_rerun")
        is False
        and sensitivity.get("strategy_ranking_rule")
        == "sharpe_descending_then_net_cagr_descending_then_strategy_id_ascending"
        and sensitivity.get("final_thesis_conclusion_status")
        == "pending_stage14_inference_and_stage15_synthesis"
        and sensitivity.get("counterfactuals_are_primary_estimates") is False
        and sensitivity.get("scenario_results_may_be_promoted") is False,
        "terminal-event sensitivity contract differs",
    )
    return policy_id, scope, treatment, monthly_rule, sensitivity


def identify_last_tradable_close_liquidations(
    *,
    delistings: pa.Table,
    returns: pa.Table,
    market: pa.Table,
    monthly_reconciliation: pa.Table,
    policy: Mapping[str, object],
    policy_sha256: str,
) -> LastTradableCloseOverlay:
    """Authenticate the exact Stage 13 last-tradable-close population.

    The input source tables remain unchanged.  The caller applies the returned
    events only to final accounting rows and binds the receipt into the
    replacement Stage 12/Stage 13 identity.
    """

    policy_id, scope, treatment, monthly_rule, sensitivity = _policy_identity(
        policy, policy_sha256
    )
    _required_columns(delistings, _DELISTING_COLUMNS, label="delisting evidence")
    _required_columns(returns, _RETURN_COLUMNS, label="daily return evidence")
    _required_columns(market, _MARKET_COLUMNS, label="daily market evidence")
    _required_columns(
        monthly_reconciliation,
        _MONTHLY_COLUMNS,
        label="monthly return evidence",
    )

    structural = tuple(
        MappingProxyType(dict(row))
        for row in delistings.select(_DELISTING_COLUMNS).to_pylist()
        if _structural_event_candidate(row, policy=policy)
    )
    _require(structural, "terminal-event structural population is empty")
    permnos = pa.array(sorted({int(row["permno"]) for row in structural}), pa.int64())
    routed_returns = returns.select(_RETURN_COLUMNS).filter(
        pc.is_in(returns["permno"], value_set=permnos)
    )
    routed_market = market.select(_MARKET_COLUMNS).filter(
        pc.is_in(market["permno"], value_set=permnos)
    )
    routed_monthly = monthly_reconciliation.select(_MONTHLY_COLUMNS).filter(
        pc.is_in(monthly_reconciliation["permno"], value_set=permnos)
    )
    return_rows = _unique_rows_by_key(
        routed_returns,
        key_names=("permno", "calendar_date"),
        label="terminal-event return evidence",
    )
    market_rows = _unique_rows_by_key(
        routed_market,
        key_names=("permno", "calendar_date"),
        label="terminal-event market evidence",
    )
    monthly_rows = _unique_rows_by_key(
        routed_monthly,
        key_names=("permno", "calendar_month"),
        label="terminal-event monthly evidence",
    )

    session_rule = policy["final_session_eligibility"]
    assert isinstance(session_rule, Mapping)
    matches: list[LastTradableCloseEvent] = []
    rejected_structural_keys: list[list[object]] = []
    for event in structural:
        permno = int(event["permno"])
        event_date = _date(event["delisting_date"], label="delisting date")
        posting_date = _date(event["posting_date"], label="posting date")
        final_return = return_rows.get((permno, event_date))
        final_market = market_rows.get((permno, event_date))
        posting = return_rows.get((permno, posting_date))
        if final_return is None or final_market is None or posting is None:
            rejected_structural_keys.append([permno, event_date.isoformat()])
            continue
        raw_return = final_return["raw_total_return"]
        raw_retx = final_return["raw_return_ex_distributions"]
        if not (
            isinstance(raw_return, Decimal)
            and raw_return.is_finite()
            and raw_return >= Decimal(str(session_rule["raw_total_return_minimum"]))
            and raw_retx == raw_return
            and final_return["raw_return_missing_flag"]
            == session_rule["raw_total_return_missing_flag"]
            and final_return["mapped_trading_periods"]
            == session_rule["mapped_trading_periods"]
            and final_return["return_leg_class"] == session_rule["return_leg_class"]
            and final_return["covered_trading_sessions"]
            == session_rule["covered_trading_sessions"]
            and final_return["span_alignment_status"]
            == session_rule["span_alignment_status"]
            and final_return["distribution_event_count"] == 0
            and final_return["delisting_event_count"] == 1
            and final_return["delisting_treatment_status"]
            == policy["event_eligibility"]["treatment_status"]
            and final_return["delisting_adjustment_return"] is None
            and final_return["authoritative_total_return"] is None
            and final_return["authoritative_return_state"]
            == "unresolved_nonperformance_delisting"
            and final_return["return_history_usable"] is False
            and final_return["return_history_authorized"] is False
            and final_return["outcome_availability_date"] == posting_date
        ):
            rejected_structural_keys.append([permno, event_date.isoformat()])
            continue
        span_start = final_return["return_span_start_date"]
        if (
            type(span_start) is not dt.date
            or final_return["return_span_end_date"] != event_date
        ):
            rejected_structural_keys.append([permno, event_date.isoformat()])
            continue
        previous_market = market_rows.get((permno, span_start))
        event_price = event["delisting_price"]
        close = final_market["close_usd_per_share"]
        bid = final_market["bid_usd_per_share"]
        ask = final_market["ask_usd_per_share"]
        previous_close = (
            None if previous_market is None else previous_market["close_usd_per_share"]
        )
        if not (
            final_market["conditional_type"] == session_rule["conditional_type"]
            and final_market["source_trading_status_flag"]
            == session_rule["trading_status_flag"]
            and final_market["price_flag"] == session_rule["price_flag"]
            and final_market["screening_price_valid"] is True
            and final_market["liquidity_session_valid"] is True
            and type(final_market["volume_shares"]) is int
            and final_market["volume_shares"] > 0
            and isinstance(close, Decimal)
            and close.is_finite()
            and close > 0
            and close == event_price
            and isinstance(bid, Decimal)
            and bid.is_finite()
            and bid > 0
            and isinstance(ask, Decimal)
            and ask.is_finite()
            and ask >= bid
            and isinstance(previous_close, Decimal)
            and previous_close.is_finite()
            and previous_close > 0
        ):
            rejected_structural_keys.append([permno, event_date.isoformat()])
            continue
        with localcontext() as context:
            context.prec = 50
            quoted_full_spread = ((ask - bid) / ((ask + bid) / Decimal(2))).quantize(
                Decimal("0.000000000000000001")
            )
        if not quoted_full_spread.is_finite() or quoted_full_spread > Decimal("0.20"):
            rejected_structural_keys.append([permno, event_date.isoformat()])
            continue
        reproduced = close / previous_close - Decimal(1)
        reproduction_tolerance = Decimal(
            str(session_rule["maximum_close_reproduction_absolute_difference"])
        )
        if abs(reproduced - raw_return) > reproduction_tolerance:
            rejected_structural_keys.append([permno, event_date.isoformat()])
            continue
        later_market_rows = [
            row
            for (row_permno, row_date), row in market_rows.items()
            if row_permno == permno
            and isinstance(row_date, dt.date)
            and row_date > event_date
        ]
        later_tracked_market = [
            row
            for row in later_market_rows
            if row["close_usd_per_share"] is not None
            or row["bid_usd_per_share"] is not None
            or row["ask_usd_per_share"] is not None
            or row["volume_shares"] is not None
            or row["source_trading_status_flag"] == "A"
            or row["conditional_type"] == "RW"
            or row["screening_price_valid"] is True
            or row["liquidity_session_valid"] is True
        ]
        posting_market_tombstone = (
            len(later_market_rows) == 1
            and later_market_rows[0]["calendar_date"] == posting_date
            and later_market_rows[0]["source_trading_status_flag"] == "D"
            and later_market_rows[0]["conditional_type"] == "N/A"
            and later_market_rows[0]["price_flag"] == "DM"
            and later_market_rows[0]["close_usd_per_share"] is None
            and later_market_rows[0]["bid_usd_per_share"] is None
            and later_market_rows[0]["ask_usd_per_share"] is None
            and later_market_rows[0]["volume_shares"] is None
            and later_market_rows[0]["screening_price_valid"] is False
            and later_market_rows[0]["liquidity_session_valid"] is False
        )
        later_returns = sorted(
            (
                row
                for (row_permno, row_date), row in return_rows.items()
                if row_permno == permno
                and isinstance(row_date, dt.date)
                and row_date > event_date
            ),
            key=lambda row: row["calendar_date"],
        )
        posting_valid = (
            len(later_returns) == 1
            and later_returns[0]["calendar_date"] == posting_date
            and posting["raw_total_return"] is None
            and posting["raw_return_ex_distributions"] is None
            and posting["raw_return_missing_flag"]
            == session_rule["following_posting_return_missing_flag"]
            and posting["raw_return_duration_flag"]
            == session_rule["following_posting_return_duration_flag"]
            and posting["mapped_trading_periods"] is None
            and posting["return_leg_class"]
            == "terminal_delisting_posting_audit_suppressed"
            and posting["return_span_start_date"] == event_date
            and posting["return_span_end_date"] == posting_date
            and posting["covered_trading_sessions"] == 0
            and posting["delisting_treatment_status"]
            == "missing_nonperformance_terminal_leg_unresolved_posting_payload_suppressed"
            and posting["authoritative_total_return"] is None
            and posting["return_history_usable"] is False
            and posting["return_history_authorized"] is False
        )
        if later_tracked_market or not posting_market_tombstone or not posting_valid:
            rejected_structural_keys.append([permno, event_date.isoformat()])
            continue
        monthly_candidates = [
            row
            for (row_permno, month_date), row in monthly_rows.items()
            if row_permno == permno
            and isinstance(month_date, dt.date)
            and month_date.year == event_date.year
            and month_date.month == event_date.month
        ]
        if len(monthly_candidates) != 1:
            rejected_structural_keys.append([permno, event_date.isoformat()])
            continue
        monthly = monthly_candidates[0]
        monthly_difference = monthly["numeric_transport_absolute_difference"]
        locked_tolerance = Decimal(str(monthly_rule["maximum_absolute_difference"]))
        monthly_valid = (
            monthly["raw_monthly_return_flag"]
            == monthly_rule["raw_monthly_return_flag"]
            and monthly["monthly_flag_class"] == monthly_rule["monthly_flag_class"]
            and monthly["daily_coverage_class"] == monthly_rule["daily_coverage_class"]
            and monthly["numeric_transport_status"]
            == monthly_rule["numeric_transport_status"]
            and monthly["authorized_return_diagnostic_status"]
            == "nonperformance_terminal_outcome_unresolved"
            and monthly["reconciliation_status"] == "accounted"
            and monthly["missing_daily_return_count"] == 1
            and monthly["delisting_event_count"] == 1
            and isinstance(monthly_difference, Decimal)
            and monthly_difference.is_finite()
            and monthly_difference <= locked_tolerance
            and monthly["tolerance_used"] == locked_tolerance
        )
        if not monthly_valid:
            rejected_structural_keys.append([permno, event_date.isoformat()])
            continue
        evidence = {
            "policy_id": policy_id,
            "policy_sha256": policy_sha256,
            "event": event,
            "final_return": final_return,
            "final_market": final_market,
            "previous_market": previous_market,
            "posting_return": posting,
            "monthly_reconciliation": monthly,
            "later_market_row_count": len(later_market_rows),
            "later_tracked_market_row_count": len(later_tracked_market),
            "reproduced_return": reproduced,
            "return_reproduction_absolute_difference": abs(reproduced - raw_return),
        }
        matches.append(
            LastTradableCloseEvent(
                permno=permno,
                delisting_date=event_date,
                posting_date=posting_date,
                outcome_availability_date=_date(
                    event["outcome_availability_date"],
                    label="terminal-event outcome availability date",
                ),
                ordinary_total_return=raw_return.quantize(Decimal("0.000000000001")),
                event_day_bid_usd_per_share=bid,
                event_day_ask_usd_per_share=ask,
                event_day_quoted_full_spread=quoted_full_spread,
                source_event_sha256=_sha256(event),
                evidence_sha256=_sha256(evidence),
            )
        )

    matches.sort(key=lambda event: (event.delisting_date, event.permno))
    expected_raw = scope.get("expected_matches")
    _require(isinstance(expected_raw, list), "terminal-event expected matches differ")
    _require(
        len(expected_raw) == int(scope.get("expected_match_count", -1))
        and all(isinstance(row, Mapping) for row in expected_raw),
        "terminal-event configured expected rows differ",
    )
    expected = tuple(
        sorted(
            (
                int(row["permno"]),
                dt.date.fromisoformat(str(row["delisting_date"])),
                dt.date.fromisoformat(str(row["posting_date"])),
                Decimal(str(row["event_day_bid_usd_per_share"])),
                Decimal(str(row["event_day_ask_usd_per_share"])),
                Decimal(str(row["event_day_quoted_full_spread"])),
            )
            for row in expected_raw
        )
    )
    observed = tuple(
        sorted(
            (
                event.permno,
                event.delisting_date,
                event.posting_date,
                event.event_day_bid_usd_per_share,
                event.event_day_ask_usd_per_share,
                event.event_day_quoted_full_spread,
            )
            for event in matches
        )
    )
    _require(
        len(expected) == int(scope.get("expected_match_count", -1)),
        "terminal-event configured expected count differs",
    )
    _require(
        observed == expected,
        "terminal-event final match population differs from the frozen expectation: "
        f"observed={observed!r}, expected={expected!r}, "
        f"rejected_structural_keys={rejected_structural_keys!r}",
    )
    event_receipts = [event.as_dict() for event in matches]
    receipt_payload: dict[str, object] = {
        "schema_version": 1,
        "status": "STAGE13_LAST_TRADABLE_CLOSE_LIQUIDATION_OVERLAY_AUTHENTICATED",
        "policy_id": policy_id,
        "policy_sha256": policy_sha256,
        "policy_content_sha256": _sha256(policy),
        "treatment_status": treatment["treatment_status"],
        "structural_candidate_count": len(structural),
        "qualifying_event_count": len(matches),
        "rejected_structural_keys": rejected_structural_keys,
        "events": event_receipts,
        "vendor_missing_delisting_return_preserved": True,
        "ordinary_final_session_return_authorized": True,
        "cash_retirement_multiplier": 1.0,
        "sensitivity_status": sensitivity["status"],
        "sensitivity_scenario_count": 4,
        "primary_estimates_modified_by_sensitivity": False,
        "unexpected_or_missing_match_disposition": "fail_closed",
    }
    receipt_payload["overlay_sha256"] = _sha256(receipt_payload)
    return LastTradableCloseOverlay(
        policy_id=policy_id,
        policy_sha256=policy_sha256,
        treatment_status=str(treatment["treatment_status"]),
        events=tuple(matches),
        receipt=MappingProxyType(receipt_payload),
    )


__all__ = [
    "LastTradableCloseEvent",
    "LastTradableCloseOverlay",
    "TerminalEventOverlayError",
    "identify_last_tradable_close_liquidations",
]
