"""Deterministic Stage 6 feature registries and categorical vocabularies.

The helpers in this module build only the three processed registry tables
specified by the frozen Stage 6 contract.  They do not write files, inspect
validation/final-period data, construct model matrices, or publish artifacts.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import tomllib
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import pyarrow as pa

from portfolio_bot.features.feature_store_contract import (
    DEFAULT_POLICY,
    PROJECT_ROOT,
    expanded_features,
    validate_feature_store_contract,
)


FEATURE_DICTIONARY_SCHEMA = pa.schema(
    [
        pa.field("feature_ordinal", pa.int16(), nullable=False),
        pa.field("feature_name", pa.string(), nullable=False),
        pa.field("family", pa.string(), nullable=False),
        pa.field("kind", pa.string(), nullable=False),
        pa.field("source_set", pa.string(), nullable=False),
        pa.field("unit", pa.string(), nullable=False),
        pa.field("formula_id", pa.string(), nullable=False),
        pa.field("parameters_json", pa.string(), nullable=False),
    ]
)
FEATURE_SET_MEMBERSHIP_SCHEMA = pa.schema(
    [
        pa.field("feature_set", pa.string(), nullable=False),
        pa.field("feature_ordinal", pa.int16(), nullable=False),
        pa.field("feature_name", pa.string(), nullable=False),
        pa.field("included", pa.bool_(), nullable=False),
    ]
)
CATEGORY_VOCABULARY_SCHEMA = pa.schema(
    [
        pa.field("feature_name", pa.string(), nullable=False),
        pa.field("category_ordinal", pa.int32(), nullable=False),
        pa.field("category_value", pa.string(), nullable=False),
        pa.field("encoded_column_name", pa.string(), nullable=False),
        pa.field("is_unknown", pa.bool_(), nullable=False),
        pa.field("is_no_event", pa.bool_(), nullable=False),
        pa.field("population", pa.string(), nullable=False),
    ]
)

POPULATION_FIXED_REGISTRY = "FROZEN_REGISTRY_ENUMERATION"
POPULATION_FIXED_FF12 = "FROZEN_FF12_MAPPING"
POPULATION_RESERVED = "FROZEN_RESERVED_CATEGORY"
POPULATION_DEVELOPMENT = "LONG_ELIGIBLE_DEVELOPMENT"

_FEATURE_FIELDS = {
    "name",
    "family",
    "kind",
    "source_set",
    "unit",
    "formula_id",
}
_CATEGORICAL_KINDS = {"categorical_firm", "categorical_multihot_firm"}
_SIC2 = re.compile(r"[0-9]{2}\Z")
_COUNTRY = re.compile(r"[A-Z]{3}\Z")
_SLUG_RUN = re.compile(r"[^a-z0-9]+")


class FeatureStoreRegistryError(RuntimeError):
    """Raised when a processed Stage 6 registry cannot be frozen safely."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FeatureStoreRegistryError(message)


def _project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _normalize_category(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = unicodedata.normalize("NFKC", value).strip().upper()
    return normalized or None


def category_slug(category_value: str) -> str:
    """Return the frozen ASCII slug for one exact normalized category."""

    _require(
        isinstance(category_value, str) and bool(category_value),
        "Category slug input must be a nonempty string",
    )
    decomposed = unicodedata.normalize("NFKD", category_value)
    ascii_text = "".join(
        character
        for character in decomposed
        if ord(character) < 128 and not unicodedata.combining(character)
    ).lower()
    slug = _SLUG_RUN.sub("_", ascii_text).strip("_")
    return slug or "category"


def encoded_category_column_name(feature_name: str, category_value: str) -> str:
    """Return the collision-resistant frozen one-hot column name."""

    _require(
        isinstance(feature_name, str) and bool(feature_name),
        "Feature name must be a nonempty string",
    )
    _require(
        isinstance(category_value, str) and bool(category_value),
        "Category value must be a nonempty string",
    )
    suffix = hashlib.sha256(category_value.encode("utf-8")).hexdigest()[:8]
    return (
        f"{feature_name}__category__{category_slug(category_value)}__{suffix}"
    )


@dataclass(frozen=True)
class _RegistryContext:
    features: tuple[Mapping[str, Any], ...]
    feature_sets: Mapping[str, tuple[str, ...]]
    ff12_categories: tuple[str, ...]
    first_signal_date: dt.date
    last_signal_date: dt.date


def _load_context(policy_path: Path) -> _RegistryContext:
    policy_path = policy_path.resolve()
    validated = validate_feature_store_contract(policy_path)
    with policy_path.open("rb") as handle:
        policy = tomllib.load(handle)
    registry_path = _project_path(policy["semantic_contract"]["registry_path"])
    registry = json.loads(registry_path.read_text("utf-8"))
    _require(isinstance(registry, dict), "Frozen feature registry is not an object")
    features = tuple(expanded_features(registry))
    _require(
        len(features) == validated["feature_count"] == 163,
        "Frozen feature registry count changed",
    )
    return _RegistryContext(
        features=features,
        feature_sets=validated["feature_sets"],
        ff12_categories=tuple(validated["ff12_categories"]),
        first_signal_date=dt.date.fromisoformat(
            str(policy["scope"]["first_signal_date"])
        ),
        last_signal_date=dt.date.fromisoformat(
            str(policy["scope"]["last_signal_date"])
        ),
    )


def build_feature_dictionary(
    policy_path: Path = DEFAULT_POLICY,
) -> pa.Table:
    """Build the exact 163-row physical feature dictionary."""

    context = _load_context(policy_path)
    rows: list[dict[str, Any]] = []
    for ordinal, feature in enumerate(context.features, start=1):
        parameters = {
            key: value
            for key, value in feature.items()
            if key not in _FEATURE_FIELDS
        }
        rows.append(
            {
                "feature_ordinal": ordinal,
                "feature_name": str(feature["name"]),
                "family": str(feature["family"]),
                "kind": str(feature["kind"]),
                "source_set": str(feature["source_set"]),
                "unit": str(feature["unit"]),
                "formula_id": str(feature["formula_id"]),
                "parameters_json": json.dumps(
                    parameters,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
            }
        )
    table = pa.Table.from_pylist(rows, schema=FEATURE_DICTIONARY_SCHEMA)
    _require(table.num_rows == 163, "Feature dictionary is not 163 rows")
    return table


def build_feature_set_membership(
    policy_path: Path = DEFAULT_POLICY,
) -> pa.Table:
    """Build the complete three-set by 163-feature Boolean membership matrix."""

    context = _load_context(policy_path)
    feature_ordinals = {
        str(feature["name"]): ordinal
        for ordinal, feature in enumerate(context.features, start=1)
    }
    rows: list[dict[str, Any]] = []
    for feature_set in sorted(context.feature_sets, key=lambda value: value.encode("utf-8")):
        included = set(context.feature_sets[feature_set])
        for feature in context.features:
            name = str(feature["name"])
            rows.append(
                {
                    "feature_set": feature_set,
                    "feature_ordinal": feature_ordinals[name],
                    "feature_name": name,
                    "included": name in included,
                }
            )
    table = pa.Table.from_pylist(rows, schema=FEATURE_SET_MEMBERSHIP_SCHEMA)
    _require(table.num_rows == 489, "Feature-set membership is not 489 rows")
    return table


def _unknown_category(feature: Mapping[str, Any]) -> str:
    explicit = _normalize_category(feature.get("unknown"))
    if explicit is not None:
        return explicit
    categories = feature.get("categories")
    if isinstance(categories, list):
        unknowns = [
            normalized
            for value in categories
            if (normalized := _normalize_category(value)) is not None
            and normalized.startswith("UNKNOWN")
        ]
        _require(
            len(unknowns) == 1,
            f"Fixed category feature has no unique unknown: {feature['name']}",
        )
        return unknowns[0]
    _require(
        feature["kind"] == "categorical_multihot_firm",
        f"Dynamic categorical feature has no explicit unknown: {feature['name']}",
    )
    return "UNKNOWN"


def _fixed_categories(
    feature: Mapping[str, Any],
    ff12_categories: tuple[str, ...],
) -> tuple[tuple[str, str], ...] | None:
    categories = feature.get("categories")
    if isinstance(categories, list):
        normalized = tuple(_normalize_category(value) for value in categories)
        _require(
            all(value is not None for value in normalized),
            f"Fixed category is blank: {feature['name']}",
        )
        values = tuple(str(value) for value in normalized)
        _require(
            len(values) == len(set(values)),
            f"Fixed categories are duplicated: {feature['name']}",
        )
        return tuple(
            (value, POPULATION_FIXED_REGISTRY) for value in values
        )
    if feature["name"] == "ff12_sector_category":
        _require(
            len(ff12_categories) == 13
            and len(ff12_categories) == len(set(ff12_categories)),
            "FF12 category enumeration changed",
        )
        return tuple(
            (value, POPULATION_FIXED_FF12) for value in ff12_categories
        )
    return None


def _observed_values(
    row: Mapping[str, Any],
    feature: Mapping[str, Any],
) -> tuple[str, ...]:
    name = str(feature["name"])
    value = row.get(name)
    if feature["kind"] == "categorical_firm":
        normalized = _normalize_category(value)
        _require(normalized is not None, f"Invalid categorical value: {name}")
        return (normalized,)
    _require(
        isinstance(value, (list, tuple)) and bool(value),
        f"Invalid multihot value: {name}",
    )
    normalized_values = tuple(_normalize_category(item) for item in value)
    _require(
        all(item is not None for item in normalized_values),
        f"Invalid multihot category: {name}",
    )
    result = tuple(dict.fromkeys(str(item) for item in normalized_values))
    reserved = {"UNKNOWN", "NO_EVENT"}.intersection(result)
    _require(
        not reserved or len(result) == 1,
        f"Reserved multihot category was combined with another value: {name}",
    )
    return result


def _validate_dynamic_category(name: str, value: str) -> None:
    if name == "sic2_industry_category":
        _require(
            value == "UNKNOWN_INDUSTRY" or _SIC2.fullmatch(value) is not None,
            "Invalid SIC2 category entered the development vocabulary",
        )
    elif name == "domicile_country_category":
        _require(
            value in {"UNKNOWN_COUNTRY", "UNKNOWN_FOREIGN"}
            or _COUNTRY.fullmatch(value) is not None,
            "Invalid domicile category entered the development vocabulary",
        )


def build_category_vocabulary(
    rows: Iterable[Mapping[str, Any]],
    policy_path: Path = DEFAULT_POLICY,
) -> pa.Table:
    """Freeze categorical values from long-eligible development rows only.

    Registry and FF12 enumerations are included in full even when unobserved.
    Dynamic categories are normalized and collected only from rows marked
    ``long_eligible=True`` inside the frozen 2000-01-31 through 2014-10-31
    development signal range.
    """

    context = _load_context(policy_path)
    categorical = tuple(
        feature
        for feature in context.features
        if feature["kind"] in _CATEGORICAL_KINDS
    )
    fixed = {
        str(feature["name"]): _fixed_categories(
            feature, context.ff12_categories
        )
        for feature in categorical
    }
    dynamic = {
        str(feature["name"]): set()
        for feature in categorical
        if fixed[str(feature["name"])] is None
    }
    row_count = long_count = 0
    for row in rows:
        row_count += 1
        _require(isinstance(row, Mapping), "Category source row is not a mapping")
        signal_date = row.get("signal_date")
        _require(
            isinstance(signal_date, dt.date)
            and not isinstance(signal_date, dt.datetime),
            "Category source signal_date is not a date",
        )
        _require(
            context.first_signal_date
            <= signal_date
            <= context.last_signal_date,
            "Category source row escaped the frozen development period",
        )
        long_eligible = row.get("long_eligible")
        _require(
            isinstance(long_eligible, bool),
            "Category source long_eligible is not Boolean",
        )
        if not long_eligible:
            continue
        long_count += 1
        for feature in categorical:
            name = str(feature["name"])
            observed = _observed_values(row, feature)
            fixed_values = fixed[name]
            if fixed_values is not None:
                allowed = {value for value, _population in fixed_values}
                _require(
                    set(observed).issubset(allowed),
                    f"Observed category escaped frozen enumeration: {name}",
                )
                continue
            unknown = _unknown_category(feature)
            reserved = {unknown}
            if feature["kind"] == "categorical_multihot_firm":
                reserved.add("NO_EVENT")
            for value in observed:
                _validate_dynamic_category(name, value)
                if value not in reserved:
                    dynamic[name].add(value)
    _require(row_count > 0, "Category vocabulary source is empty")
    _require(
        long_count > 0,
        "Category vocabulary has no long-eligible development rows",
    )

    vocabulary_rows: list[dict[str, Any]] = []
    for feature in sorted(
        categorical,
        key=lambda item: str(item["name"]).encode("utf-8"),
    ):
        name = str(feature["name"])
        unknown = _unknown_category(feature)
        categories = fixed[name]
        if categories is None:
            reserved = [(unknown, POPULATION_RESERVED)]
            if feature["kind"] == "categorical_multihot_firm":
                reserved.append(("NO_EVENT", POPULATION_RESERVED))
            observed = [
                (value, POPULATION_DEVELOPMENT)
                for value in sorted(
                    dynamic[name],
                    key=lambda item: item.encode("utf-8"),
                )
            ]
            categories = tuple((*reserved, *observed))
        _require(
            len(categories) == len({value for value, _population in categories}),
            f"Category vocabulary contains duplicates: {name}",
        )
        for ordinal, (value, population) in enumerate(categories, start=1):
            vocabulary_rows.append(
                {
                    "feature_name": name,
                    "category_ordinal": ordinal,
                    "category_value": value,
                    "encoded_column_name": encoded_category_column_name(
                        name, value
                    ),
                    "is_unknown": value == unknown,
                    "is_no_event": value == "NO_EVENT",
                    "population": population,
                }
            )
    encoded = [row["encoded_column_name"] for row in vocabulary_rows]
    _require(
        len(encoded) == len(set(encoded)),
        "Encoded categorical column names are not unique",
    )
    return pa.Table.from_pylist(
        vocabulary_rows,
        schema=CATEGORY_VOCABULARY_SCHEMA,
    )
