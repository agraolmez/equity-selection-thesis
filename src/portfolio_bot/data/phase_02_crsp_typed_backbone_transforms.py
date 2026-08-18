"""Vectorized Arrow transforms for the locked Phase 2 CRSP typed backbone.

The public functions accept retained, canonical raw UTF-8 batches.  Arrow
compute kernels perform validation, parsing, and derivation without converting
columns to Python objects.  Only an empty numeric/date lexeme becomes null;
malformed, non-exact, or out-of-range nonblank values are fatal.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Iterable, Mapping

import pyarrow as pa
import pyarrow.compute as pc

from portfolio_bot.data.phase_02_crsp_typed_backbone_core import (
    TableSpec,
    TypedBackboneCoreError,
    parse_arrow_type,
)


class TypedBackboneTransformError(ValueError):
    """Raised when a canonical raw row violates the locked typed contract."""


ArrowColumn = pa.Array | pa.ChunkedArray

_DECIMAL_PATTERN = r"^[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[Ee][+-]?[0-9]+)?$"
_DATE_PATTERN = r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$"
_POSITIVE_INTEGER_PATTERN = r"^[1-9][0-9]*$"
_NONNEGATIVE_INTEGER_PATTERN = r"^[0-9]+$"
_YYYYMM_PATTERN = r"^[0-9]{6}$"

_DURATION_PERIODS: Mapping[str, int] = {
    "D1": 1,
    "D2": 1,
    "D3": 1,
    "D4": 1,
    "DU": 1,
    "P1": 2,
    "P2": 3,
    "P3": 4,
    "P4": 5,
    "P5": 6,
    "P6": 7,
    "P7": 8,
    "P8": 9,
    "P9": 10,
    "MR": 0,
    "DD": -1,
}

# Bind every accepted spec to the locked field order, types, nullability,
# metadata, keys, and expected production row count.
_LOCKED_TABLE_FINGERPRINTS: Mapping[str, str] = {
    "crsp_security_day": (
        "dd5277f2a106d5b3c2f487bfdcad6694895c2f77ad51fe455bba9a2342e223de"
    ),
    "crsp_security_month_crosscheck": (
        "c97988ea872244f65ce9f172473d8a3d2d116027cec9670cb8be56b105da8027"
    ),
    "crsp_identity_interval": (
        "f5a0952fd6d91e39952220d058ec915d7bf62a4f4936917a31da7c6610766f17"
    ),
    "crsp_shares_interval": (
        "80b436af3685cf0d185ffce0f110c901d8a9bfe06e19b72b2da88f405252bab8"
    ),
    "crsp_security_day_return_staging": (
        "929482b0e2658538f4ccc18829290f76a124314bc60a7aefad6cbc3f865d6e2f"
    ),
    "crsp_security_month_return_staging": (
        "18a6ffd971162ff190cbc4f07bbde2c9718fbf16e9d57e2e9ee0702b4197a615"
    ),
}

_DAILY_RAW_COLUMNS = frozenset(
    {
        "PERMNO",
        "PERMCO",
        "SecInfoStartDt",
        "SecInfoEndDt",
        "CUSIP",
        "PrimaryExch",
        "ConditionalType",
        "TradingStatusFlg",
        "USIncFlg",
        "IssuerType",
        "SecurityType",
        "SecuritySubType",
        "ShareType",
        "Ticker",
        "SICCD",
        "NAICS",
        "ICBIndustry",
        "DlyCalDt",
        "DlyPrc",
        "DlyPrcFlg",
        "DlyCap",
        "DlyVol",
        "DlyClose",
        "DlyLow",
        "DlyHigh",
        "DlyBid",
        "DlyAsk",
        "DlyOpen",
        "DlyPrcVol",
        "DlyRet",
        "DlyRetx",
        "DlyRetMissFlg",
        "DlyRetDurFlg",
    }
)

_MONTHLY_RAW_COLUMNS = frozenset(
    {
        "PERMNO",
        "PERMCO",
        "SecInfoStartDt",
        "SecInfoEndDt",
        "CUSIP",
        "PrimaryExch",
        "ConditionalType",
        "TradingStatusFlg",
        "USIncFlg",
        "IssuerType",
        "SecurityType",
        "SecuritySubType",
        "ShareType",
        "SICCD",
        "YYYYMM",
        "MthCalDt",
        "MthPrc",
        "MthPrcFlg",
        "MthPrcDt",
        "MthCap",
        "MthDisCnt",
        "MthVol",
        "MthVolFlg",
        "MthFacShrFlg",
        "MthPrcVol",
        "MthPrcVolMissCnt",
        "ShrOut",
        "MthRet",
        "MthRetx",
        "MthRetFlg",
    }
)

_IDENTITY_RAW_COLUMNS = frozenset(
    {
        "PERMNO",
        "PERMCO",
        "SecInfoStartDt",
        "SecInfoEndDt",
        "SecurityBegDt",
        "SecurityEndDt",
        "CUSIP",
        "Ticker",
        "IssuerNm",
        "USIncFlg",
        "IssuerType",
        "SecurityType",
        "SecuritySubType",
        "ShareType",
        "SICCD",
        "PrimaryExch",
        "NAICS",
        "TradingStatusFlg",
    }
)

_SHARES_RAW_COLUMNS = frozenset(
    {
        "PERMNO",
        "ShrStartDt",
        "ShrEndDt",
        "ShrOut",
        "ShrSource",
        "ShrFacType",
        "ShrAdrFlg",
    }
)


def _require_locked_spec(spec: TableSpec, expected_name: str) -> None:
    if type(spec) is not TableSpec:
        raise TypedBackboneTransformError(
            f"{expected_name} spec must use the exact TableSpec type"
        )
    if spec.name != expected_name:
        raise TypedBackboneTransformError(
            f"Expected {expected_name} spec, received {spec.name!r}"
        )
    expected = _LOCKED_TABLE_FINGERPRINTS[expected_name]
    try:
        observed = spec.fingerprint()
    except TypedBackboneCoreError as exc:
        raise TypedBackboneTransformError(
            f"Cannot authenticate {expected_name} schema"
        ) from exc
    if observed != expected:
        raise TypedBackboneTransformError(
            f"{expected_name} spec differs from the locked schema"
        )


def _raw_strings(
    raw_batch: pa.RecordBatch | pa.Table,
    required: Iterable[str],
) -> tuple[dict[str, ArrowColumn], int]:
    if not isinstance(raw_batch, (pa.RecordBatch, pa.Table)):
        raise TypedBackboneTransformError(
            "raw_batch must be an Arrow RecordBatch or Table"
        )
    names = raw_batch.schema.names
    if len(names) != len(set(names)):
        raise TypedBackboneTransformError("Raw input column names must be unique")
    missing = set(required) - set(names)
    if missing:
        raise TypedBackboneTransformError(
            f"Raw input is missing required columns: {sorted(missing)}"
        )

    columns: dict[str, ArrowColumn] = {}
    for index, name in enumerate(names):
        column = raw_batch.column(index)
        if column.type != pa.string():
            raise TypedBackboneTransformError(
                f"Raw column {name!r} must be exact Arrow UTF-8, not {column.type}"
            )
        if column.null_count:
            raise TypedBackboneTransformError(
                f"Raw column {name!r} contains a non-string or Arrow null"
            )
        columns[name] = column
    return columns, raw_batch.num_rows


def _require_all(
    mask: ArrowColumn,
    message: str,
    *,
    source: ArrowColumn | None = None,
) -> None:
    """Raise on the first false/null mask value without materializing a column."""

    if len(mask) == 0:
        return
    safe_mask = pc.fill_null(mask, False)
    result = pc.all(safe_mask, skip_nulls=False)
    if result.is_valid and result.as_py() is True:
        return
    row = pc.index(safe_mask, False).as_py()
    suffix = f" at canonical row {row}"
    if source is not None and row is not None and row >= 0:
        suffix += f": {source[row].as_py()!r}"
    raise TypedBackboneTransformError(message + suffix)


def _blank_mask(raw: ArrowColumn) -> ArrowColumn:
    return pc.equal(raw, pa.scalar("", type=pa.string()))


def _nullable_strings(raw: ArrowColumn, blank: ArrowColumn) -> ArrowColumn:
    return pc.if_else(blank, pa.scalar(None, type=pa.string()), raw)


def _positive_int64(
    raw: ArrowColumn,
    label: str,
    *,
    optional: bool = False,
) -> ArrowColumn:
    blank = _blank_mask(raw)
    if not optional:
        _require_all(pc.invert(blank), f"{label} is blank", source=raw)
    grammar = pc.match_substring_regex(raw, _POSITIVE_INTEGER_PATTERN)
    accepted = pc.or_(blank, grammar) if optional else grammar
    _require_all(
        accepted,
        f"{label} is not a positive ASCII integer",
        source=raw,
    )
    prepared = _nullable_strings(raw, blank) if optional else raw
    try:
        return pc.cast(prepared, pa.int64(), safe=True)
    except pa.ArrowException as exc:
        raise TypedBackboneTransformError(f"{label} exceeds signed int64") from exc


def _nonnegative_int64(raw: ArrowColumn, label: str) -> ArrowColumn:
    blank = _blank_mask(raw)
    grammar = pc.match_substring_regex(raw, _NONNEGATIVE_INTEGER_PATTERN)
    _require_all(
        pc.or_(blank, grammar),
        f"{label} is not a nonnegative ASCII integer",
        source=raw,
    )
    try:
        return pc.cast(_nullable_strings(raw, blank), pa.int64(), safe=True)
    except pa.ArrowException as exc:
        raise TypedBackboneTransformError(f"{label} exceeds signed int64") from exc


def _date32(
    raw: ArrowColumn,
    label: str,
    *,
    optional: bool = False,
) -> ArrowColumn:
    blank = _blank_mask(raw)
    if not optional:
        _require_all(pc.invert(blank), f"{label} is blank", source=raw)
    grammar = pc.match_substring_regex(raw, _DATE_PATTERN)
    accepted = pc.or_(blank, grammar) if optional else grammar
    _require_all(accepted, f"{label} is not strict YYYY-MM-DD", source=raw)

    year = pc.utf8_slice_codeunits(raw, start=0, stop=4)
    positive_year = pc.greater(year, pa.scalar("0000", type=pa.string()))
    accepted_year = pc.or_(blank, positive_year) if optional else positive_year
    _require_all(accepted_year, f"{label} is not a Gregorian date", source=raw)

    prepared = _nullable_strings(raw, blank) if optional else raw
    try:
        return pc.cast(prepared, pa.date32(), safe=True)
    except pa.ArrowException as exc:
        raise TypedBackboneTransformError(f"{label} is not a Gregorian date") from exc


def _decimal128_exact(
    raw: ArrowColumn,
    data_type: pa.Decimal128Type,
    label: str,
) -> ArrowColumn:
    """Vector-validate a decimal lexeme before Arrow's decimal128 cast.

    Arrow 19's direct string-to-decimal cast detects scale loss but can wrap a
    coefficient wider than 128 bits.  The lexical arithmetic below proves the
    target unscaled coefficient has at most ``precision`` digits and that any
    discarded source digits are zero before invoking the cast.
    """

    blank = _blank_mask(raw)
    grammar = pc.match_substring_regex(raw, _DECIMAL_PATTERN)
    _require_all(
        pc.or_(blank, grammar),
        f"{label} is not a decimal lexeme",
        source=raw,
    )

    zero = pa.scalar("0", type=pa.string())
    analysis = pc.if_else(blank, zero, raw)
    mantissa = pc.replace_substring_regex(analysis, pattern=r"[Ee].*$", replacement="")
    unsigned_mantissa = pc.replace_substring_regex(
        mantissa, pattern=r"^[+-]", replacement=""
    )
    fraction = pc.replace_substring_regex(
        unsigned_mantissa, pattern=r"^[^.]*\.?", replacement=""
    )
    coefficient = pc.replace_substring_regex(
        unsigned_mantissa, pattern=r"\.", replacement=""
    )
    significant = pc.replace_substring_regex(
        coefficient, pattern=r"^0+", replacement=""
    )
    significant_length = pc.cast(pc.utf8_length(significant), pa.int64())
    is_zero = pc.equal(significant_length, pa.scalar(0, type=pa.int64()))

    has_exponent = pc.match_substring_regex(analysis, r"[Ee]")
    extracted_exponent = pc.replace_substring_regex(
        analysis, pattern=r"^.*[Ee]", replacement=""
    )
    exponent_lexeme = pc.if_else(has_exponent, extracted_exponent, zero)
    negative_exponent = pc.starts_with(exponent_lexeme, pattern="-")
    exponent_digits = pc.replace_substring_regex(
        exponent_lexeme, pattern=r"^[+-]", replacement=""
    )
    exponent_significant = pc.replace_substring_regex(
        exponent_digits, pattern=r"^0+", replacement=""
    )
    exponent_length = pc.utf8_length(exponent_significant)

    # A nonzero value cannot be representable when |exponent| exceeds the
    # maximum number of code units in an Arrow UTF-8 cell.  This bound also
    # makes the following integer cast unconditionally safe.
    exponent_size_ok = pc.or_(
        pc.less(exponent_length, pa.scalar(10, type=pa.int32())),
        pc.and_(
            pc.equal(exponent_length, pa.scalar(10, type=pa.int32())),
            pc.less_equal(
                exponent_significant,
                pa.scalar("2147483647", type=pa.string()),
            ),
        ),
    )
    _require_all(
        pc.or_(is_zero, exponent_size_ok),
        f"{label} is not exactly representable as {data_type}",
        source=raw,
    )

    normalized_digits = pc.if_else(
        pc.equal(exponent_length, pa.scalar(0, type=pa.int32())),
        zero,
        exponent_significant,
    )
    exponent_sign = pc.if_else(
        negative_exponent,
        pa.scalar("-", type=pa.string()),
        pa.scalar("", type=pa.string()),
    )
    normalized_exponent = pc.binary_join_element_wise(
        exponent_sign, normalized_digits, ""
    )
    safe_exponent = pc.if_else(is_zero, zero, normalized_exponent)
    exponent = pc.cast(safe_exponent, pa.int64(), safe=True)

    fraction_length = pc.cast(pc.utf8_length(fraction), pa.int64())
    without_trailing_zeros = pc.replace_substring_regex(
        coefficient, pattern=r"0+$", replacement=""
    )
    trailing_zeros = pc.subtract_checked(
        pc.cast(pc.utf8_length(coefficient), pa.int64()),
        pc.cast(pc.utf8_length(without_trailing_zeros), pa.int64()),
    )
    shift = pc.add_checked(
        pc.subtract_checked(exponent, fraction_length),
        pa.scalar(data_type.scale, type=pa.int64()),
    )
    nonnegative_shift = pc.greater_equal(shift, pa.scalar(0, type=pa.int64()))
    scale_exact = pc.or_(
        nonnegative_shift,
        pc.greater_equal(trailing_zeros, pc.negate_checked(shift)),
    )
    unscaled_digits = pc.add_checked(significant_length, shift)
    precision_ok = pc.and_(
        pc.greater(unscaled_digits, pa.scalar(0, type=pa.int64())),
        pc.less_equal(
            unscaled_digits,
            pa.scalar(data_type.precision, type=pa.int64()),
        ),
    )
    representable = pc.or_(
        blank,
        pc.or_(is_zero, pc.and_(scale_exact, precision_ok)),
    )
    _require_all(
        representable,
        f"{label} is not exactly representable as {data_type}",
        source=raw,
    )

    # Canonicalize every proven nonzero value before the Arrow cast.  Casting
    # the original lexeme is unsafe in Arrow 19: a coefficient wider than
    # 128 bits can wrap even when its exponent makes the numeric value small.
    # Removing trailing zeros keeps the coefficient short whenever such an
    # exponent cancellation is valid.
    normalized_coefficient = pc.replace_substring_regex(
        significant, pattern=r"0+$", replacement=""
    )
    effective_exponent = pc.add_checked(
        pc.subtract_checked(exponent, fraction_length),
        trailing_zeros,
    )
    value_negative = pc.starts_with(mantissa, pattern="-")
    value_sign = pc.if_else(
        value_negative,
        pa.scalar("-", type=pa.string()),
        pa.scalar("", type=pa.string()),
    )
    canonical_nonzero = pc.binary_join_element_wise(
        value_sign,
        normalized_coefficient,
        "e",
        pc.cast(effective_exponent, pa.string(), safe=True),
        "",
    )
    prepared = pc.if_else(
        blank,
        pa.scalar(None, type=pa.string()),
        pc.if_else(is_zero, zero, canonical_nonzero),
    )
    options = pc.CastOptions(
        target_type=data_type,
        allow_decimal_truncate=False,
    )
    try:
        return pc.cast(prepared, options=options)
    except pa.ArrowException as exc:
        raise TypedBackboneTransformError(
            f"{label} is not exactly representable as {data_type}"
        ) from exc


def _scale_decimal_1000(
    values: ArrowColumn,
    data_type: pa.Decimal128Type,
    label: str,
) -> ArrowColumn:
    wider_type = pa.decimal256(data_type.precision, data_type.scale)
    multiplier_type = pa.decimal256(4, 0)
    wider = pc.cast(values, wider_type, safe=True)
    result = pc.multiply_checked(
        wider,
        pa.scalar(Decimal("1000"), type=multiplier_type),
    )
    try:
        return pc.cast(
            result,
            options=pc.CastOptions(
                target_type=data_type,
                allow_decimal_truncate=False,
            ),
        )
    except pa.ArrowException as exc:
        raise TypedBackboneTransformError(
            f"{label} is not exactly representable as {data_type}"
        ) from exc


def _positive_decimal(values: ArrowColumn) -> ArrowColumn:
    zero = pa.scalar(Decimal("0"), type=values.type)
    return pc.fill_null(pc.greater(values, zero), False)


def _dollar_volume(
    close: ArrowColumn,
    volume: ArrowColumn,
    valid: ArrowColumn,
    data_type: pa.Decimal128Type,
) -> ArrowColumn:
    masked_close = pc.if_else(valid, close, pa.scalar(None, type=close.type))
    masked_volume = pc.if_else(valid, volume, pa.scalar(None, type=pa.int64()))
    close_wide = pc.cast(
        masked_close,
        pa.decimal256(data_type.precision, data_type.scale),
        safe=True,
    )
    volume_wide = pc.cast(masked_volume, pa.decimal256(19, 0), safe=True)
    result = pc.multiply_checked(close_wide, volume_wide)
    try:
        return pc.cast(
            result,
            options=pc.CastOptions(
                target_type=data_type,
                allow_decimal_truncate=False,
            ),
        )
    except pa.ArrowException as exc:
        raise TypedBackboneTransformError(
            "screening_dollar_volume_usd is not exactly representable as "
            f"{data_type}"
        ) from exc


def _assert_ordered(
    starts: ArrowColumn,
    ends: ArrowColumn,
    label: str,
) -> None:
    _require_all(pc.less_equal(starts, ends), f"{label} is inverted")


def _duration_mapping(raw: ArrowColumn) -> ArrowColumn:
    keys = pa.array(tuple(_DURATION_PERIODS), type=pa.string())
    values = pa.array(tuple(_DURATION_PERIODS.values()), type=pa.int16())
    indices = pc.index_in(raw, value_set=keys)
    known = pc.greater_equal(indices, pa.scalar(0, type=indices.type))
    safe_indices = pc.if_else(
        known,
        indices,
        pa.scalar(0, type=indices.type),
    )
    mapped = pc.take(values, safe_indices)
    return pc.if_else(known, mapped, pa.scalar(None, type=pa.int16()))


def _constant(value: object, data_type: pa.DataType, rows: int) -> pa.Array:
    return pa.repeat(pa.scalar(value, type=data_type), rows)


def _build_table(
    spec: TableSpec,
    columns: Mapping[str, ArrowColumn],
    row_count: int,
) -> pa.Table:
    field_names = tuple(field.name for field in spec.fields)
    if set(columns) != set(field_names):
        missing = set(field_names) - set(columns)
        extra = set(columns) - set(field_names)
        raise TypedBackboneTransformError(
            f"Output fields differ for {spec.name}; missing={sorted(missing)}, "
            f"extra={sorted(extra)}"
        )

    arrays: list[ArrowColumn] = []
    for field in spec.fields:
        values = columns[field.name]
        expected_type = parse_arrow_type(field.logical_type)
        if len(values) != row_count:
            raise TypedBackboneTransformError(
                f"Output length differs for {spec.name}.{field.name}"
            )
        if values.type != expected_type:
            raise TypedBackboneTransformError(
                f"Output type differs for {spec.name}.{field.name}: "
                f"{values.type} != {expected_type}"
            )
        if not field.nullable and values.null_count:
            raise TypedBackboneTransformError(
                f"Nonnullable output {spec.name}.{field.name} contains null"
            )
        arrays.append(values)

    expected_schema = spec.arrow_schema()
    try:
        table = pa.Table.from_arrays(arrays, schema=expected_schema)
        table.validate(full=True)
    except (pa.ArrowException, OverflowError, TypeError, ValueError) as exc:
        raise TypedBackboneTransformError(
            f"Cannot construct locked output {spec.name}"
        ) from exc
    if not table.schema.equals(expected_schema, check_metadata=True):
        raise TypedBackboneTransformError(
            f"Constructed {spec.name} schema differs from the locked schema"
        )
    return table


def transform_daily(
    raw_batch: pa.RecordBatch | pa.Table,
    consumer_spec: TableSpec,
    return_spec: TableSpec,
) -> tuple[pa.Table, pa.Table]:
    """Type one canonical daily batch and split consumer/return payloads."""

    _require_locked_spec(consumer_spec, "crsp_security_day")
    _require_locked_spec(return_spec, "crsp_security_day_return_staging")
    raw, rows = _raw_strings(raw_batch, _DAILY_RAW_COLUMNS)

    decimal_6 = pa.decimal128(38, 6)
    decimal_2 = pa.decimal128(38, 2)
    decimal_1 = pa.decimal128(38, 1)
    permno = _positive_int64(raw["PERMNO"], "daily PERMNO")
    permco = _positive_int64(raw["PERMCO"], "daily PERMCO", optional=True)
    interval_start = _date32(raw["SecInfoStartDt"], "daily security-information start")
    interval_end = _date32(raw["SecInfoEndDt"], "daily security-information end")
    _assert_ordered(
        interval_start,
        interval_end,
        "Daily security-information interval",
    )
    calendar_date = _date32(raw["DlyCalDt"], "daily calendar date")
    diagnostic_price = _decimal128_exact(raw["DlyPrc"], decimal_6, "DlyPrc")
    market_cap = _decimal128_exact(raw["DlyCap"], decimal_2, "DlyCap")
    volume = _nonnegative_int64(raw["DlyVol"], "DlyVol")
    close = _decimal128_exact(raw["DlyClose"], decimal_6, "DlyClose")
    low = _decimal128_exact(raw["DlyLow"], decimal_6, "DlyLow")
    high = _decimal128_exact(raw["DlyHigh"], decimal_6, "DlyHigh")
    bid = _decimal128_exact(raw["DlyBid"], decimal_6, "DlyBid")
    ask = _decimal128_exact(raw["DlyAsk"], decimal_6, "DlyAsk")
    opened = _decimal128_exact(raw["DlyOpen"], decimal_6, "DlyOpen")
    vendor_price_volume = _decimal128_exact(raw["DlyPrcVol"], decimal_1, "DlyPrcVol")
    normalized_cap = _scale_decimal_1000(market_cap, decimal_2, "market_cap_usd")
    price_valid = _positive_decimal(close)
    market_cap_valid = _positive_decimal(market_cap)
    liquidity_valid = pc.and_(price_valid, pc.is_valid(volume))
    dollar_volume = _dollar_volume(
        close,
        volume,
        liquidity_valid,
        decimal_6,
    )

    consumer = {
        "permno": permno,
        "source_permco": permco,
        "source_security_info_start_date": interval_start,
        "source_security_info_end_date": interval_end,
        "source_cusip": raw["CUSIP"],
        "source_primary_exchange": raw["PrimaryExch"],
        "conditional_type": raw["ConditionalType"],
        "source_trading_status_flag": raw["TradingStatusFlg"],
        "source_us_incorporation_flag": raw["USIncFlg"],
        "source_issuer_type": raw["IssuerType"],
        "source_security_type": raw["SecurityType"],
        "source_security_subtype": raw["SecuritySubType"],
        "source_share_type": raw["ShareType"],
        "source_ticker": raw["Ticker"],
        "source_sic_code": raw["SICCD"],
        "source_naics_code": raw["NAICS"],
        "source_icb_industry": raw["ICBIndustry"],
        "calendar_date": calendar_date,
        "diagnostic_price_usd_per_share": diagnostic_price,
        "price_flag": raw["DlyPrcFlg"],
        "market_cap_thousand_usd": market_cap,
        "market_cap_usd": normalized_cap,
        "volume_shares": volume,
        "close_usd_per_share": close,
        "low_usd_per_share": low,
        "high_usd_per_share": high,
        "bid_usd_per_share": bid,
        "ask_usd_per_share": ask,
        "open_usd_per_share": opened,
        "vendor_price_volume_usd": vendor_price_volume,
        "screening_dollar_volume_usd": dollar_volume,
        "screening_price_valid": price_valid,
        "market_cap_valid": market_cap_valid,
        "liquidity_session_valid": liquidity_valid,
    }

    durations = raw["DlyRetDurFlg"]
    returns = {
        "permno": permno,
        "calendar_date": calendar_date,
        "raw_total_return": _decimal128_exact(raw["DlyRet"], decimal_6, "DlyRet"),
        "raw_return_ex_distributions": _decimal128_exact(
            raw["DlyRetx"], decimal_6, "DlyRetx"
        ),
        "raw_return_missing_flag": raw["DlyRetMissFlg"],
        "raw_return_duration_flag": durations,
        "mapped_trading_periods": _duration_mapping(durations),
        "reconciliation_status": _constant("unreconciled", pa.string(), rows),
        "return_history_authorized": _constant(False, pa.bool_(), rows),
    }
    return (
        _build_table(consumer_spec, consumer, rows),
        _build_table(return_spec, returns, rows),
    )


def transform_monthly(
    raw_batch: pa.RecordBatch | pa.Table,
    consumer_spec: TableSpec,
    return_spec: TableSpec,
) -> tuple[pa.Table, pa.Table]:
    """Type one canonical monthly batch and split consumer/return payloads."""

    _require_locked_spec(consumer_spec, "crsp_security_month_crosscheck")
    _require_locked_spec(return_spec, "crsp_security_month_return_staging")
    raw, rows = _raw_strings(raw_batch, _MONTHLY_RAW_COLUMNS)

    decimal_6 = pa.decimal128(38, 6)
    decimal_2 = pa.decimal128(38, 2)
    decimal_1 = pa.decimal128(38, 1)
    permno = _positive_int64(raw["PERMNO"], "monthly PERMNO")
    permco = _positive_int64(raw["PERMCO"], "monthly PERMCO", optional=True)
    interval_start = _date32(
        raw["SecInfoStartDt"], "monthly security-information start"
    )
    interval_end = _date32(raw["SecInfoEndDt"], "monthly security-information end")
    _assert_ordered(
        interval_start,
        interval_end,
        "Monthly security-information interval",
    )
    calendar_month = _date32(raw["MthCalDt"], "monthly calendar date")
    observation_date = _date32(
        raw["MthPrcDt"], "monthly observation date", optional=True
    )

    yyyymm_raw = raw["YYYYMM"]
    _require_all(
        pc.match_substring_regex(yyyymm_raw, _YYYYMM_PATTERN),
        "YYYYMM is not six ASCII digits",
        source=yyyymm_raw,
    )
    calendar_period = pc.binary_join_element_wise(
        pc.utf8_slice_codeunits(raw["MthCalDt"], start=0, stop=4),
        pc.utf8_slice_codeunits(raw["MthCalDt"], start=5, stop=7),
        "",
    )
    _require_all(
        pc.equal(yyyymm_raw, calendar_period),
        "YYYYMM differs from MthCalDt period",
        source=yyyymm_raw,
    )
    observation_period = pc.binary_join_element_wise(
        pc.utf8_slice_codeunits(raw["MthPrcDt"], start=0, stop=4),
        pc.utf8_slice_codeunits(raw["MthPrcDt"], start=5, stop=7),
        "",
    )
    observation_blank = _blank_mask(raw["MthPrcDt"])
    _require_all(
        pc.or_(observation_blank, pc.equal(yyyymm_raw, observation_period)),
        "MthPrcDt differs from YYYYMM period",
        source=raw["MthPrcDt"],
    )
    yyyymm = pc.cast(yyyymm_raw, pa.int32(), safe=True)

    price = _decimal128_exact(raw["MthPrc"], decimal_6, "MthPrc")
    market_cap = _decimal128_exact(raw["MthCap"], decimal_2, "MthCap")
    normalized_cap = _scale_decimal_1000(
        market_cap, decimal_2, "monthly market_cap_usd"
    )
    consumer = {
        "permno": permno,
        "source_permco": permco,
        "source_security_info_start_date": interval_start,
        "source_security_info_end_date": interval_end,
        "source_cusip": raw["CUSIP"],
        "source_primary_exchange": raw["PrimaryExch"],
        "source_conditional_type": raw["ConditionalType"],
        "source_trading_status_flag": raw["TradingStatusFlg"],
        "source_us_incorporation_flag": raw["USIncFlg"],
        "source_issuer_type": raw["IssuerType"],
        "source_security_type": raw["SecurityType"],
        "source_security_subtype": raw["SecuritySubType"],
        "source_share_type": raw["ShareType"],
        "source_sic_code": raw["SICCD"],
        "yyyymm": yyyymm,
        "calendar_month": calendar_month,
        "price_usd_per_share": price,
        "price_flag": raw["MthPrcFlg"],
        "observation_date": observation_date,
        "market_cap_thousand_usd": market_cap,
        "market_cap_usd": normalized_cap,
        "distribution_count": _nonnegative_int64(raw["MthDisCnt"], "MthDisCnt"),
        "volume_shares": _nonnegative_int64(raw["MthVol"], "MthVol"),
        "volume_flag": raw["MthVolFlg"],
        "source_share_factor_flag": raw["MthFacShrFlg"],
        "price_volume_usd": _decimal128_exact(raw["MthPrcVol"], decimal_1, "MthPrcVol"),
        "price_volume_missing_count": _nonnegative_int64(
            raw["MthPrcVolMissCnt"], "MthPrcVolMissCnt"
        ),
        "embedded_shares_outstanding_thousand": _positive_int64(
            raw["ShrOut"], "monthly ShrOut", optional=True
        ),
        "observation_date_available": pc.is_valid(observation_date),
        "market_cap_valid": _positive_decimal(market_cap),
    }
    returns = {
        "permno": permno,
        "calendar_month": calendar_month,
        "raw_total_return": _decimal128_exact(raw["MthRet"], decimal_6, "MthRet"),
        "raw_return_ex_distributions": _decimal128_exact(
            raw["MthRetx"], decimal_6, "MthRetx"
        ),
        "raw_return_flag": raw["MthRetFlg"],
        "reconciliation_status": _constant("unreconciled", pa.string(), rows),
        "return_history_authorized": _constant(False, pa.bool_(), rows),
    }
    return (
        _build_table(consumer_spec, consumer, rows),
        _build_table(return_spec, returns, rows),
    )


def transform_identity(
    raw_batch: pa.RecordBatch | pa.Table,
    spec: TableSpec,
) -> pa.Table:
    """Type one canonical effective-dated identity batch."""

    _require_locked_spec(spec, "crsp_identity_interval")
    raw, rows = _raw_strings(raw_batch, _IDENTITY_RAW_COLUMNS)
    start = _date32(raw["SecInfoStartDt"], "identity interval start")
    end = _date32(raw["SecInfoEndDt"], "identity interval end")
    security_start = _date32(raw["SecurityBegDt"], "security lifetime start")
    security_end = _date32(raw["SecurityEndDt"], "security lifetime end")
    _assert_ordered(start, end, "Identity interval")
    _assert_ordered(security_start, security_end, "Security lifetime")
    columns = {
        "permno": _positive_int64(raw["PERMNO"], "identity PERMNO"),
        "permco": _positive_int64(raw["PERMCO"], "identity PERMCO"),
        "start_date": start,
        "end_date": end,
        "security_begin_date": security_start,
        "security_end_date": security_end,
        "cusip": raw["CUSIP"],
        "ticker": raw["Ticker"],
        "issuer_name": raw["IssuerNm"],
        "us_incorporation_flag": raw["USIncFlg"],
        "issuer_type": raw["IssuerType"],
        "security_type": raw["SecurityType"],
        "security_subtype": raw["SecuritySubType"],
        "share_type": raw["ShareType"],
        "sic_code": raw["SICCD"],
        "primary_exchange": raw["PrimaryExch"],
        "naics_code": raw["NAICS"],
        "trading_status_flag": raw["TradingStatusFlg"],
    }
    return _build_table(spec, columns, rows)


def transform_shares(
    raw_batch: pa.RecordBatch | pa.Table,
    spec: TableSpec,
) -> pa.Table:
    """Type one canonical effective-dated shares batch."""

    _require_locked_spec(spec, "crsp_shares_interval")
    raw, rows = _raw_strings(raw_batch, _SHARES_RAW_COLUMNS)
    start = _date32(raw["ShrStartDt"], "shares interval start")
    end = _date32(raw["ShrEndDt"], "shares interval end")
    _assert_ordered(start, end, "Shares interval")
    thousands = _positive_int64(raw["ShrOut"], "shares ShrOut")
    try:
        shares = pc.multiply_checked(
            thousands,
            pa.scalar(1000, type=pa.int64()),
        )
    except pa.ArrowException as exc:
        raise TypedBackboneTransformError(
            "shares_outstanding exceeds signed int64 after x1000"
        ) from exc
    columns = {
        "permno": _positive_int64(raw["PERMNO"], "shares PERMNO"),
        "start_date": start,
        "end_date": end,
        "shares_outstanding_thousand": thousands,
        "shares_outstanding": shares,
        "shares_source": raw["ShrSource"],
        "shares_factor_type": raw["ShrFacType"],
        "shares_adr_flag": raw["ShrAdrFlg"],
    }
    return _build_table(spec, columns, rows)


__all__ = [
    "TypedBackboneTransformError",
    "transform_daily",
    "transform_identity",
    "transform_monthly",
    "transform_shares",
]
