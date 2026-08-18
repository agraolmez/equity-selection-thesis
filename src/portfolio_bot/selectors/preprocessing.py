"""Leakage-safe, filesystem-agnostic preprocessing for Stage 9 selectors.

The module deliberately accepts already-authenticated in-memory tables.  It
does not resolve pointers, read files, fit a selector, or inspect labels beyond
the two positional masks supplied by the caller.  The frozen 551-column
encoding is retained for provenance while ``model_matrix`` contains only the
columns with non-zero support in valid outer-training rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from portfolio_bot.features.feature_math import centered_average_ranks


CONTINUOUS_FIRM = "continuous_firm"
BINARY_FIRM = "binary_firm"
AGGREGATE_STATE = "aggregate_state"
CATEGORICAL_FIRM = "categorical_firm"
CATEGORICAL_MULTIHOT_FIRM = "categorical_multihot_firm"

NUMERIC_KINDS = frozenset({CONTINUOUS_FIRM, BINARY_FIRM, AGGREGATE_STATE})
CATEGORY_KINDS = frozenset({CATEGORICAL_FIRM, CATEGORICAL_MULTIHOT_FIRM})
ALLOWED_KINDS = NUMERIC_KINDS | CATEGORY_KINDS

EXPECTED_FEATURE_KIND_COUNTS = {
    CONTINUOUS_FIRM: 122,
    BINARY_FIRM: 8,
    AGGREGATE_STATE: 23,
    CATEGORICAL_FIRM: 8,
    CATEGORICAL_MULTIHOT_FIRM: 2,
}
EXPECTED_ECONOMIC_FEATURE_COUNT = 163
EXPECTED_CATEGORY_COLUMN_COUNT = 245
EXPECTED_FROZEN_COLUMN_COUNT = 551
EXPECTED_ELASTIC_NET_INTERACTION_COUNT = 2_806


@dataclass(frozen=True)
class CrossSectionRankResult:
    """One cross-section after type-7 winsorization and centered ranking."""

    values: np.ndarray
    missing: np.ndarray
    lower_type7: float | None
    upper_type7: float | None


@dataclass(frozen=True)
class PastOnlyZScoreResult:
    """Past-only expanding z-scores and their auditable history statistics."""

    values: np.ndarray
    missing: np.ndarray
    prior_count: np.ndarray
    prior_mean: np.ndarray
    prior_std_ddof1: np.ndarray


@dataclass(frozen=True)
class SelectorColumn:
    """One column in the frozen selector encoding."""

    column_ordinal: int
    column_name: str
    source_feature_ordinal: int
    source_feature_name: str
    source_kind: str
    encoding: str
    category_ordinal: int | None = None
    category_value: str | None = None


@dataclass(frozen=True)
class SelectorMatrixSchema:
    """Ordered frozen schema, independent of any fold's active support."""

    columns: tuple[SelectorColumn, ...]

    @property
    def width(self) -> int:
        return len(self.columns)

    @property
    def column_names(self) -> tuple[str, ...]:
        return tuple(column.column_name for column in self.columns)

    def receipt(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "column_ordinal": column.column_ordinal,
                    "column_name": column.column_name,
                    "source_feature_ordinal": column.source_feature_ordinal,
                    "source_feature_name": column.source_feature_name,
                    "source_kind": column.source_kind,
                    "encoding": column.encoding,
                    "category_ordinal": column.category_ordinal,
                    "category_value": column.category_value,
                }
                for column in self.columns
            ]
        )

    def as_dictionary(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for column in self.columns:
            counts[column.encoding] = counts.get(column.encoding, 0) + 1
        return {
            "schema_version": 1,
            "frozen_column_count": self.width,
            "column_names": list(self.column_names),
            "encoding_counts": counts,
            "dtype": "float32",
        }

    def arrow_schema(self) -> Any:
        """Return the closed model-column Arrow schema without requiring Arrow at import."""

        import pyarrow as pa

        return pa.schema(
            [
                pa.field(column.column_name, pa.float32(), nullable=False)
                for column in self.columns
            ]
        )


@dataclass(frozen=True)
class SelectorPreprocessingResult:
    """Frozen encoding, active fold matrix, and inspectable transformation receipts."""

    row_keys: pd.DataFrame
    scoring_row_indices: np.ndarray
    frozen_matrix: np.ndarray
    model_matrix: np.ndarray
    schema: SelectorMatrixSchema
    support_mask: np.ndarray
    valid_outer_training_scoring_mask: np.ndarray
    feature_dictionary_receipt: pd.DataFrame
    category_vocabulary_receipt: pd.DataFrame
    category_mapping_receipt: pd.DataFrame
    support_receipt: pd.DataFrame
    aggregate_history_receipt: pd.DataFrame

    @property
    def active_column_names(self) -> tuple[str, ...]:
        return tuple(
            np.asarray(self.schema.column_names, dtype=object)[self.support_mask]
        )

    def matrix_frame(self, *, active_only: bool = True) -> pd.DataFrame:
        if active_only:
            return pd.DataFrame(self.model_matrix, columns=self.active_column_names)
        return pd.DataFrame(self.frozen_matrix, columns=self.schema.column_names)


def winsorized_centered_rank(values: Iterable[Any]) -> CrossSectionRankResult:
    """Apply type-7 1/99 winsorization then exact average-tie centered ranks.

    Non-numeric and non-finite values are missing.  Ranking uses all finite
    values supplied by the caller and the frozen formula
    ``((average_rank - 0.5) / N) - 0.5``.
    """

    numeric = pd.to_numeric(
        pd.Series(list(values), dtype=object), errors="coerce"
    ).to_numpy(dtype=np.float64)
    finite = np.isfinite(numeric)
    missing = ~finite
    ranked = np.zeros(numeric.size, dtype=np.float64)
    if not finite.any():
        return CrossSectionRankResult(ranked, missing, None, None)

    population = numeric[finite]
    lower = float(np.quantile(population, 0.01, method="linear"))
    upper = float(np.quantile(population, 0.99, method="linear"))
    clipped = np.full(numeric.shape, np.nan, dtype=np.float64)
    clipped[finite] = np.clip(population, lower, upper)
    centered = centered_average_ranks(clipped)
    ranked[finite] = centered[finite]
    return CrossSectionRankResult(ranked, missing, lower, upper)


def past_only_expanding_zscore(
    ordered_signal_dates: Sequence[Any],
    values: Iterable[Any],
    *,
    clip: float = 5.0,
    minimum_prior_observations: int = 2,
) -> PastOnlyZScoreResult:
    """Standardize each state value using strictly earlier finite months.

    Dates must be unique and strictly increasing.  The current value is added
    to history only after its score has been computed.  A score is unavailable
    when the current value is missing, fewer than two prior finite observations
    exist, or the prior sample standard deviation (``ddof=1``) is not positive.
    """

    dates = pd.to_datetime(pd.Series(list(ordered_signal_dates)), errors="raise")
    if (
        dates.isna().any()
        or dates.duplicated().any()
        or not dates.is_monotonic_increasing
    ):
        raise ValueError(
            "ordered_signal_dates must be unique, non-null, and increasing"
        )
    if clip <= 0.0 or minimum_prior_observations < 2:
        raise ValueError(
            "clip must be positive and minimum_prior_observations must be >= 2"
        )

    numeric = pd.to_numeric(
        pd.Series(list(values), dtype=object), errors="coerce"
    ).to_numpy(dtype=np.float64)
    if numeric.size != dates.size:
        raise ValueError("ordered_signal_dates and values must have equal length")
    numeric[~np.isfinite(numeric)] = np.nan

    scores = np.zeros(numeric.size, dtype=np.float64)
    missing = np.ones(numeric.size, dtype=bool)
    prior_count = np.zeros(numeric.size, dtype=np.int32)
    prior_mean = np.full(numeric.size, np.nan, dtype=np.float64)
    prior_std = np.full(numeric.size, np.nan, dtype=np.float64)
    history: list[float] = []

    for position, current in enumerate(numeric):
        prior_count[position] = len(history)
        if history:
            prior_mean[position] = float(np.mean(history, dtype=np.float64))
        if len(history) >= 2:
            prior_std[position] = float(np.std(history, ddof=1, dtype=np.float64))
        if (
            np.isfinite(current)
            and len(history) >= minimum_prior_observations
            and np.isfinite(prior_std[position])
            and prior_std[position] > 0.0
        ):
            score = (float(current) - prior_mean[position]) / prior_std[position]
            scores[position] = float(np.clip(score, -clip, clip))
            missing[position] = False
        if np.isfinite(current):
            history.append(float(current))

    return PastOnlyZScoreResult(scores, missing, prior_count, prior_mean, prior_std)


def build_selector_schema(
    feature_dictionary: Any,
    category_vocabulary: Any,
    *,
    enforce_frozen_contract: bool = True,
    permit_projected_global_ordinals: bool = False,
) -> SelectorMatrixSchema:
    """Build the exact registry-major Stage 9 column schema."""

    registry = _normalize_feature_dictionary(
        feature_dictionary,
        enforce_frozen_contract,
        permit_projected_global_ordinals=permit_projected_global_ordinals,
    )
    vocabulary = _normalize_category_vocabulary(
        category_vocabulary, registry, enforce_frozen_contract
    )
    return _schema_from_normalized(registry, vocabulary, enforce_frozen_contract)


def build_elastic_net_interaction_registry(
    schema: SelectorMatrixSchema,
    support_mask: Sequence[bool] | None = None,
) -> pd.DataFrame:
    """Return continuous-major × aggregate-minor interaction columns.

    Only numeric value columns participate.  Missingness and category columns
    are never interacted.  ``active`` is informational; all 2,806 frozen rows
    remain in the receipt.
    """

    continuous = [
        column
        for column in schema.columns
        if column.source_kind == CONTINUOUS_FIRM and column.encoding == "numeric_value"
    ]
    aggregate = [
        column
        for column in schema.columns
        if column.source_kind == AGGREGATE_STATE and column.encoding == "numeric_value"
    ]
    if support_mask is None:
        active_by_name = {column.column_name: True for column in schema.columns}
    else:
        active = _strict_boolean_array(support_mask, schema.width, "support_mask")
        active_by_name = {
            column.column_name: bool(active[position])
            for position, column in enumerate(schema.columns)
        }

    records: list[dict[str, Any]] = []
    ordinal = 1
    for firm in continuous:
        for state in aggregate:
            records.append(
                {
                    "interaction_ordinal": ordinal,
                    "firm_feature_ordinal": firm.source_feature_ordinal,
                    "firm_column": firm.column_name,
                    "aggregate_feature_ordinal": state.source_feature_ordinal,
                    "aggregate_column": state.column_name,
                    "interaction_column": f"{firm.column_name}__x__{state.column_name}",
                    "active": active_by_name[firm.column_name]
                    and active_by_name[state.column_name],
                }
            )
            ordinal += 1

    result = pd.DataFrame(records)
    if (
        schema.width == EXPECTED_FROZEN_COLUMN_COUNT
        and len(result) != EXPECTED_ELASTIC_NET_INTERACTION_COUNT
    ):
        raise ValueError("Frozen Elastic Net interaction count is not 2,806")
    return result


def build_selector_matrix(
    feature_table: Any,
    feature_dictionary: Any,
    category_vocabulary: Any,
    *,
    outer_training_mask: Sequence[bool] | str,
    valid_target_mask: Sequence[bool] | str,
    aggregate_history_table: Any | None = None,
    enforce_frozen_contract: bool = True,
    permit_projected_global_ordinals: bool = False,
) -> SelectorPreprocessingResult:
    """Transform the full Stage 6 table into one leakage-safe outer-fold matrix.

    ``feature_table`` may already be restricted to long-eligible rows, but it
    must retain every such row, including rows whose future target is invalid.
    When that table omits zero-long warm-up dates, ``aggregate_history_table``
    must provide ``signal_date`` plus all aggregate-state columns at monthly
    grain for the complete authenticated Stage 6 date spine.  It may contain
    one row per month or repeated identical security rows.  Cross-sectional
    transformations use every long-eligible row at the signal date.  Target
    validity is used only for category support and the zero-variance mask.
    """

    features = _as_frame(feature_table, "feature_table")
    registry = _normalize_feature_dictionary(
        feature_dictionary,
        enforce_frozen_contract,
        permit_projected_global_ordinals=permit_projected_global_ordinals,
    )
    vocabulary = _normalize_category_vocabulary(
        category_vocabulary, registry, enforce_frozen_contract
    )
    schema = _schema_from_normalized(registry, vocabulary, enforce_frozen_contract)
    _validate_feature_table(features, registry)
    aggregate_history = (
        features
        if aggregate_history_table is None
        else _as_frame(aggregate_history_table, "aggregate_history_table")
    )
    aggregate_names = registry.loc[
        registry["kind"] == AGGREGATE_STATE, "feature_name"
    ].tolist()
    _validate_aggregate_history_table(aggregate_history, aggregate_names)

    row_count = len(features)
    outer_training = _mask_from_argument(
        outer_training_mask, features, row_count, "outer_training_mask"
    )
    target_valid = _mask_from_argument(
        valid_target_mask, features, row_count, "valid_target_mask"
    )
    long_eligible = _strict_boolean_array(
        features["long_eligible"], row_count, "long_eligible"
    )
    scoring_indices = np.flatnonzero(long_eligible)
    if scoring_indices.size == 0:
        raise ValueError("feature_table contains no long-eligible scoring rows")
    valid_training_full = long_eligible & outer_training & target_valid
    if not valid_training_full.any():
        raise ValueError("no valid long-eligible outer-training rows were supplied")
    valid_training_scoring = valid_training_full[scoring_indices]

    normalized_dates = pd.to_datetime(features["signal_date"], errors="raise")
    aggregate_history_dates = pd.to_datetime(
        aggregate_history["signal_date"], errors="raise"
    )
    scoring_dates = normalized_dates.iloc[scoring_indices].reset_index(drop=True)
    scoring_date_groups = (
        pd.Series(np.arange(scoring_indices.size, dtype=np.int64))
        .groupby(scoring_dates, sort=False)
        .apply(lambda values: values.to_numpy(dtype=np.int64))
    )

    frozen_matrix = np.zeros(
        (scoring_indices.size, schema.width), dtype=np.float32, order="C"
    )
    column_position = {
        name: position for position, name in enumerate(schema.column_names)
    }
    category_mapping_records: list[dict[str, Any]] = []
    aggregate_history_records: list[dict[str, Any]] = []

    for feature in registry.itertuples(index=False):
        name = str(feature.feature_name)
        kind = str(feature.kind)
        if kind == CONTINUOUS_FIRM:
            raw = _numeric_vector(features.iloc[scoring_indices][name])
            transformed = np.zeros(raw.size, dtype=np.float64)
            missing = ~np.isfinite(raw)
            for positions in scoring_date_groups:
                result = winsorized_centered_rank(raw[positions])
                transformed[positions] = result.values
            frozen_matrix[:, column_position[name]] = transformed.astype(np.float32)
            frozen_matrix[:, column_position[f"{name}__missing"]] = missing.astype(
                np.float32
            )
        elif kind == BINARY_FIRM:
            raw = _numeric_vector(features.iloc[scoring_indices][name])
            finite = np.isfinite(raw)
            invalid = finite & ~np.isin(raw, (0.0, 1.0))
            if invalid.any():
                raise ValueError(f"binary feature {name!r} contains values outside 0/1")
            values = np.where(finite, raw, 0.0)
            frozen_matrix[:, column_position[name]] = values.astype(np.float32)
            frozen_matrix[:, column_position[f"{name}__missing"]] = (~finite).astype(
                np.float32
            )
        elif kind == AGGREGATE_STATE:
            month_dates, month_values = _aggregate_month_values(
                aggregate_history, aggregate_history_dates, name
            )
            _require_scoring_aggregate_agreement(
                features,
                normalized_dates,
                name,
                month_dates,
                month_values,
            )
            zscore = past_only_expanding_zscore(month_dates, month_values)
            by_date = {
                pd.Timestamp(date): (
                    float(zscore.values[index]),
                    bool(zscore.missing[index]),
                )
                for index, date in enumerate(month_dates)
            }
            values = np.asarray(
                [by_date[pd.Timestamp(date)][0] for date in scoring_dates]
            )
            missing = np.asarray(
                [by_date[pd.Timestamp(date)][1] for date in scoring_dates]
            )
            frozen_matrix[:, column_position[name]] = values.astype(np.float32)
            frozen_matrix[:, column_position[f"{name}__missing"]] = missing.astype(
                np.float32
            )
            for index, date in enumerate(month_dates):
                aggregate_history_records.append(
                    {
                        "signal_date": pd.Timestamp(date).date(),
                        "feature_ordinal": int(feature.feature_ordinal),
                        "feature_name": name,
                        "raw_value": month_values[index],
                        "zscore": float(zscore.values[index]),
                        "missing": bool(zscore.missing[index]),
                        "prior_finite_months": int(zscore.prior_count[index]),
                        "prior_mean": zscore.prior_mean[index],
                        "prior_std_ddof1": zscore.prior_std_ddof1[index],
                    }
                )
        elif kind == CATEGORICAL_FIRM:
            records = vocabulary.loc[vocabulary["feature_name"] == name]
            mapped, mapping = _map_single_categories(
                features.iloc[scoring_indices][name].tolist(),
                valid_training_scoring,
                records,
            )
            category_mapping_records.extend(mapping)
            for category in records.itertuples(index=False):
                frozen_matrix[:, column_position[str(category.encoded_column_name)]] = (
                    mapped == str(category.category_value)
                ).astype(np.float32)
        elif kind == CATEGORICAL_MULTIHOT_FIRM:
            records = vocabulary.loc[vocabulary["feature_name"] == name]
            mapped_sets, mapping = _map_multihot_categories(
                features.iloc[scoring_indices][name].tolist(),
                valid_training_scoring,
                records,
            )
            category_mapping_records.extend(mapping)
            for category in records.itertuples(index=False):
                category_value = str(category.category_value)
                frozen_matrix[:, column_position[str(category.encoded_column_name)]] = (
                    np.fromiter(
                        (category_value in values for values in mapped_sets),
                        dtype=np.float32,
                        count=len(mapped_sets),
                    )
                )
        else:  # pragma: no cover - registry validation owns this branch
            raise AssertionError(f"unhandled feature kind: {kind}")

    if not np.isfinite(frozen_matrix).all():
        raise ArithmeticError(
            "selector preprocessing produced a non-finite model value"
        )

    support_mask, support_receipt = _support_receipt(
        frozen_matrix, valid_training_scoring, schema
    )
    model_matrix = np.ascontiguousarray(
        frozen_matrix[:, support_mask], dtype=np.float32
    )
    if model_matrix.shape[1] == 0:
        raise ValueError(
            "every frozen selector column has zero outer-training variance"
        )

    key_columns = [
        column
        for column in ("signal_date", "permno", "permco")
        if column in features.columns
    ]
    row_keys = features.iloc[scoring_indices][key_columns].reset_index(drop=True).copy()
    row_keys.insert(0, "source_row_number", scoring_indices.astype(np.int64))

    return SelectorPreprocessingResult(
        row_keys=row_keys,
        scoring_row_indices=scoring_indices.astype(np.int64, copy=False),
        frozen_matrix=frozen_matrix,
        model_matrix=model_matrix,
        schema=schema,
        support_mask=support_mask,
        valid_outer_training_scoring_mask=valid_training_scoring,
        feature_dictionary_receipt=registry.copy(),
        category_vocabulary_receipt=vocabulary.copy(),
        category_mapping_receipt=pd.DataFrame(category_mapping_records),
        support_receipt=support_receipt,
        aggregate_history_receipt=pd.DataFrame(aggregate_history_records),
    )


def _as_frame(table: Any, name: str) -> pd.DataFrame:
    if isinstance(table, pd.DataFrame):
        frame = table.copy(deep=False)
    elif hasattr(table, "to_pandas"):
        frame = table.to_pandas()
    else:
        raise TypeError(f"{name} must be a pandas DataFrame or expose to_pandas()")
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"{name}.to_pandas() did not return a DataFrame")
    return frame.reset_index(drop=True)


def _normalize_feature_dictionary(
    table: Any,
    enforce_frozen_contract: bool,
    *,
    permit_projected_global_ordinals: bool = False,
) -> pd.DataFrame:
    if (
        type(enforce_frozen_contract) is not bool
        or type(permit_projected_global_ordinals) is not bool
    ):
        raise TypeError("feature_dictionary contract flags must be booleans")
    if enforce_frozen_contract and permit_projected_global_ordinals:
        raise ValueError(
            "projected global ordinals cannot relax the full frozen contract"
        )
    registry = _as_frame(table, "feature_dictionary")
    required = {"feature_ordinal", "feature_name", "kind"}
    missing = required - set(registry.columns)
    if missing:
        raise ValueError(f"feature_dictionary is missing columns: {sorted(missing)}")
    registry = registry.copy()
    registry["feature_ordinal"] = pd.to_numeric(
        registry["feature_ordinal"], errors="raise"
    ).astype(np.int64)
    registry["feature_name"] = registry["feature_name"].astype(str)
    registry["kind"] = registry["kind"].astype(str)
    registry = registry.sort_values("feature_ordinal", kind="mergesort").reset_index(
        drop=True
    )
    if registry.empty:
        raise ValueError("feature_dictionary is empty")
    if (
        registry["feature_ordinal"].duplicated().any()
        or registry["feature_name"].duplicated().any()
    ):
        raise ValueError("feature_dictionary ordinals and names must be unique")
    ordinals = registry["feature_ordinal"].tolist()
    if ordinals != list(range(1, len(registry) + 1)):
        if not permit_projected_global_ordinals or any(
            value < 1 or value > EXPECTED_ECONOMIC_FEATURE_COUNT for value in ordinals
        ):
            raise ValueError(
                "feature_dictionary ordinals must be contiguous and one-based"
            )
    invalid_kinds = sorted(set(registry["kind"]) - ALLOWED_KINDS)
    if invalid_kinds:
        raise ValueError(
            f"feature_dictionary contains unsupported kinds: {invalid_kinds}"
        )
    if enforce_frozen_contract:
        counts = registry["kind"].value_counts().to_dict()
        if len(registry) != EXPECTED_ECONOMIC_FEATURE_COUNT or any(
            counts.get(kind, 0) != expected
            for kind, expected in EXPECTED_FEATURE_KIND_COUNTS.items()
        ):
            raise ValueError(
                "feature_dictionary does not match the frozen 163-feature kind counts"
            )
    return registry


def _normalize_category_vocabulary(
    table: Any, registry: pd.DataFrame, enforce_frozen_contract: bool
) -> pd.DataFrame:
    vocabulary = _as_frame(table, "category_vocabulary")
    required = {
        "feature_name",
        "category_ordinal",
        "category_value",
        "encoded_column_name",
        "is_unknown",
        "is_no_event",
    }
    missing = required - set(vocabulary.columns)
    if missing:
        raise ValueError(f"category_vocabulary is missing columns: {sorted(missing)}")
    vocabulary = vocabulary.copy()
    vocabulary["feature_name"] = vocabulary["feature_name"].astype(str)
    vocabulary["category_value"] = vocabulary["category_value"].astype(str)
    vocabulary["encoded_column_name"] = vocabulary["encoded_column_name"].astype(str)
    vocabulary["category_ordinal"] = pd.to_numeric(
        vocabulary["category_ordinal"], errors="raise"
    ).astype(np.int64)
    for column in ("is_unknown", "is_no_event"):
        vocabulary[column] = _strict_boolean_array(
            vocabulary[column], len(vocabulary), f"category_vocabulary.{column}"
        )
    registry_order = dict(zip(registry["feature_name"], registry["feature_ordinal"]))
    vocabulary["_feature_ordinal"] = vocabulary["feature_name"].map(registry_order)
    if vocabulary["_feature_ordinal"].isna().any():
        raise ValueError(
            "category_vocabulary references a feature outside the dictionary"
        )
    category_names = set(
        registry.loc[registry["kind"].isin(CATEGORY_KINDS), "feature_name"]
    )
    if set(vocabulary["feature_name"]) != category_names:
        raise ValueError(
            "category_vocabulary must cover exactly the categorical features"
        )
    vocabulary = vocabulary.sort_values(
        ["_feature_ordinal", "category_ordinal"], kind="mergesort"
    ).reset_index(drop=True)
    if vocabulary.duplicated(["feature_name", "category_ordinal"]).any():
        raise ValueError("category ordinals must be unique within feature")
    if vocabulary.duplicated(["feature_name", "category_value"]).any():
        raise ValueError("category values must be unique within feature")
    if vocabulary["encoded_column_name"].duplicated().any():
        raise ValueError("encoded category column names must be globally unique")

    kind_by_name = dict(zip(registry["feature_name"], registry["kind"]))
    for feature_name, group in vocabulary.groupby("feature_name", sort=False):
        if group["category_ordinal"].tolist() != list(range(1, len(group) + 1)):
            raise ValueError(
                f"category ordinals for {feature_name!r} are not contiguous"
            )
        if int(group["is_unknown"].sum()) != 1:
            raise ValueError(f"{feature_name!r} must have exactly one UNKNOWN category")
        expected_no_event = (
            1 if kind_by_name[feature_name] == CATEGORICAL_MULTIHOT_FIRM else 0
        )
        if int(group["is_no_event"].sum()) != expected_no_event:
            raise ValueError(
                f"{feature_name!r} has the wrong number of explicit no-event categories"
            )
    if enforce_frozen_contract and len(vocabulary) != EXPECTED_CATEGORY_COLUMN_COUNT:
        raise ValueError("category_vocabulary does not contain the frozen 245 columns")
    return vocabulary.drop(columns="_feature_ordinal")


def _schema_from_normalized(
    registry: pd.DataFrame, vocabulary: pd.DataFrame, enforce_frozen_contract: bool
) -> SelectorMatrixSchema:
    columns: list[SelectorColumn] = []
    ordinal = 1
    for feature in registry.itertuples(index=False):
        feature_ordinal = int(feature.feature_ordinal)
        name = str(feature.feature_name)
        kind = str(feature.kind)
        if kind in NUMERIC_KINDS:
            columns.append(
                SelectorColumn(
                    ordinal,
                    name,
                    feature_ordinal,
                    name,
                    kind,
                    "numeric_value",
                )
            )
            ordinal += 1
            columns.append(
                SelectorColumn(
                    ordinal,
                    f"{name}__missing",
                    feature_ordinal,
                    name,
                    kind,
                    "missing_indicator",
                )
            )
            ordinal += 1
        else:
            encoding = "one_hot" if kind == CATEGORICAL_FIRM else "multi_hot"
            for category in vocabulary.loc[
                vocabulary["feature_name"] == name
            ].itertuples(index=False):
                columns.append(
                    SelectorColumn(
                        ordinal,
                        str(category.encoded_column_name),
                        feature_ordinal,
                        name,
                        kind,
                        encoding,
                        int(category.category_ordinal),
                        str(category.category_value),
                    )
                )
                ordinal += 1
    names = [column.column_name for column in columns]
    if len(names) != len(set(names)):
        raise ValueError("frozen selector output column names are not unique")
    schema = SelectorMatrixSchema(tuple(columns))
    if enforce_frozen_contract and schema.width != EXPECTED_FROZEN_COLUMN_COUNT:
        raise ValueError("selector schema width is not the frozen 551 columns")
    return schema


def _validate_feature_table(features: pd.DataFrame, registry: pd.DataFrame) -> None:
    required = {"signal_date", "long_eligible", *registry["feature_name"].tolist()}
    missing = required - set(features.columns)
    if missing:
        raise ValueError(f"feature_table is missing columns: {sorted(missing)}")
    dates = pd.to_datetime(features["signal_date"], errors="raise")
    if dates.isna().any():
        raise ValueError("feature_table contains a null signal_date")
    if (
        "permno" in features.columns
        and features.duplicated(["signal_date", "permno"]).any()
    ):
        raise ValueError("feature_table contains duplicate (signal_date, permno) keys")


def _validate_aggregate_history_table(
    aggregate_history: pd.DataFrame, aggregate_feature_names: Sequence[str]
) -> None:
    required = {"signal_date", *aggregate_feature_names}
    missing = required - set(aggregate_history.columns)
    if missing:
        raise ValueError(
            f"aggregate_history_table is missing columns: {sorted(missing)}"
        )
    dates = pd.to_datetime(aggregate_history["signal_date"], errors="raise")
    if dates.isna().any():
        raise ValueError("aggregate_history_table contains a null signal_date")


def _mask_from_argument(
    mask: Sequence[bool] | str,
    features: pd.DataFrame,
    expected_length: int,
    name: str,
) -> np.ndarray:
    if isinstance(mask, str) and mask not in features.columns:
        raise ValueError(f"{name} column {mask!r} is absent")
    values = features[mask] if isinstance(mask, str) else mask
    return _strict_boolean_array(values, expected_length, name)


def _strict_boolean_array(values: Any, expected_length: int, name: str) -> np.ndarray:
    series = pd.Series(values).reset_index(drop=True)
    if len(series) != expected_length:
        raise ValueError(f"{name} has length {len(series)}, expected {expected_length}")
    if series.isna().any():
        raise ValueError(f"{name} contains missing values")
    valid = series.isin([True, False, 0, 1])
    if not bool(valid.all()):
        raise ValueError(f"{name} must contain only boolean/0/1 values")
    return series.astype(bool).to_numpy(dtype=bool)


def _numeric_vector(values: Any) -> np.ndarray:
    result = pd.to_numeric(
        pd.Series(values).reset_index(drop=True), errors="coerce"
    ).to_numpy(dtype=np.float64)
    result[~np.isfinite(result)] = np.nan
    return result


def _aggregate_month_values(
    features: pd.DataFrame, normalized_dates: pd.Series, feature_name: str
) -> tuple[list[pd.Timestamp], np.ndarray]:
    raw = _numeric_vector(features[feature_name])
    dates = sorted(pd.Timestamp(value) for value in normalized_dates.unique())
    values = np.full(len(dates), np.nan, dtype=np.float64)
    normalized_array = normalized_dates.to_numpy()
    for position, date in enumerate(dates):
        month = raw[normalized_array == date.to_datetime64()]
        finite = month[np.isfinite(month)]
        if finite.size and finite.size != month.size:
            raise ValueError(
                f"aggregate feature {feature_name!r} mixes missing and present values on {date.date()}"
            )
        unique = np.unique(finite)
        if unique.size > 1:
            raise ValueError(
                f"aggregate feature {feature_name!r} is not common within {date.date()}"
            )
        if unique.size == 1:
            values[position] = float(unique[0])
    return dates, values


def _require_scoring_aggregate_agreement(
    features: pd.DataFrame,
    normalized_dates: pd.Series,
    feature_name: str,
    history_dates: Sequence[pd.Timestamp],
    history_values: np.ndarray,
) -> None:
    scoring_dates, scoring_values = _aggregate_month_values(
        features, normalized_dates, feature_name
    )
    history = {
        pd.Timestamp(date): float(value) if np.isfinite(value) else np.nan
        for date, value in zip(history_dates, history_values)
    }
    missing_dates = [
        date.date() for date in scoring_dates if pd.Timestamp(date) not in history
    ]
    if missing_dates:
        raise ValueError(
            f"aggregate_history_table omits scoring dates for {feature_name!r}: {missing_dates[:5]}"
        )
    for date, scoring_value in zip(scoring_dates, scoring_values):
        history_value = history[pd.Timestamp(date)]
        both_missing = not np.isfinite(scoring_value) and not np.isfinite(history_value)
        if not both_missing and not (
            np.isfinite(scoring_value)
            and np.isfinite(history_value)
            and float(scoring_value) == float(history_value)
        ):
            raise ValueError(
                f"aggregate history disagrees with scoring rows for {feature_name!r} "
                f"on {pd.Timestamp(date).date()}"
            )


def _is_scalar_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (list, tuple, set, np.ndarray)):
        return False
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    return bool(missing) if isinstance(missing, (bool, np.bool_)) else False


def _scalar_category(value: Any) -> str | None:
    return None if _is_scalar_missing(value) else str(value)


def _multihot_tokens(value: Any) -> tuple[str, ...] | None:
    if _is_scalar_missing(value):
        return None
    if isinstance(value, np.ndarray):
        items = value.tolist()
    elif isinstance(value, (list, tuple, set)):
        items = list(value)
    else:
        items = [value]
    tokens = tuple(str(item) for item in items if not _is_scalar_missing(item))
    return tokens


def _map_single_categories(
    values: list[Any], training_mask: np.ndarray, vocabulary: pd.DataFrame
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    unknown = str(vocabulary.loc[vocabulary["is_unknown"], "category_value"].iloc[0])
    frozen_values = set(vocabulary["category_value"].astype(str))
    normalized = [_scalar_category(value) for value in values]
    seen_counts: dict[str, int] = {}
    for value, included in zip(normalized, training_mask):
        if not included:
            continue
        category = value if value in frozen_values else unknown
        category = unknown if category is None else category
        seen_counts[category] = seen_counts.get(category, 0) + 1
    seen = set(seen_counts)
    mapped = np.asarray(
        [
            value if value in frozen_values and value in seen else unknown
            for value in normalized
        ],
        dtype=object,
    )
    mapping = _category_mapping_records(vocabulary, seen_counts, mapped)
    return mapped, mapping


def _map_multihot_categories(
    values: list[Any], training_mask: np.ndarray, vocabulary: pd.DataFrame
) -> tuple[list[frozenset[str]], list[dict[str, Any]]]:
    unknown = str(vocabulary.loc[vocabulary["is_unknown"], "category_value"].iloc[0])
    no_event = str(vocabulary.loc[vocabulary["is_no_event"], "category_value"].iloc[0])
    frozen_values = set(vocabulary["category_value"].astype(str))
    normalized = [_multihot_tokens(value) for value in values]
    seen_counts: dict[str, int] = {}
    for tokens, included in zip(normalized, training_mask):
        if not included:
            continue
        if tokens is None:
            mapped_training = {unknown}
        elif len(tokens) == 0:
            mapped_training = {no_event}
        else:
            mapped_training = {
                token if token in frozen_values else unknown for token in tokens
            }
        for category in mapped_training:
            seen_counts[category] = seen_counts.get(category, 0) + 1
    seen = set(seen_counts)

    mapped: list[frozenset[str]] = []
    for tokens in normalized:
        if tokens is None:
            categories = {unknown}
        elif len(tokens) == 0:
            categories = {no_event if no_event in seen else unknown}
        else:
            categories = {
                token if token in frozen_values and token in seen else unknown
                for token in tokens
            }
        mapped.append(frozenset(categories))
    mapping = _category_mapping_records(vocabulary, seen_counts, mapped)
    return mapped, mapping


def _category_mapping_records(
    vocabulary: pd.DataFrame,
    seen_counts: dict[str, int],
    mapped_values: Sequence[Any],
) -> list[dict[str, Any]]:
    unknown = str(vocabulary.loc[vocabulary["is_unknown"], "category_value"].iloc[0])
    output_counts: dict[str, int] = {}
    for value in mapped_values:
        categories = value if isinstance(value, (set, frozenset)) else (value,)
        for category in categories:
            output_counts[str(category)] = output_counts.get(str(category), 0) + 1
    records: list[dict[str, Any]] = []
    for category in vocabulary.itertuples(index=False):
        value = str(category.category_value)
        seen = value in seen_counts
        records.append(
            {
                "feature_name": str(category.feature_name),
                "category_ordinal": int(category.category_ordinal),
                "category_value": value,
                "encoded_column_name": str(category.encoded_column_name),
                "is_unknown": bool(category.is_unknown),
                "is_no_event": bool(category.is_no_event),
                "seen_in_valid_outer_training": seen,
                "valid_outer_training_row_count": int(seen_counts.get(value, 0)),
                "mapped_to_category_value": (
                    value if seen or bool(category.is_unknown) else unknown
                ),
                "encoded_scoring_row_count": int(output_counts.get(value, 0)),
            }
        )
    return records


def _support_receipt(
    matrix: np.ndarray,
    valid_training_mask: np.ndarray,
    schema: SelectorMatrixSchema,
) -> tuple[np.ndarray, pd.DataFrame]:
    training = matrix[valid_training_mask].astype(np.float64, copy=False)
    minimum = np.min(training, axis=0)
    maximum = np.max(training, axis=0)
    active = maximum > minimum
    nonzero = np.count_nonzero(training, axis=0)
    records: list[dict[str, Any]] = []
    for position, column in enumerate(schema.columns):
        records.append(
            {
                "column_ordinal": column.column_ordinal,
                "column_name": column.column_name,
                "source_feature_ordinal": column.source_feature_ordinal,
                "source_feature_name": column.source_feature_name,
                "source_kind": column.source_kind,
                "encoding": column.encoding,
                "valid_outer_training_rows": int(training.shape[0]),
                "training_nonzero_rows": int(nonzero[position]),
                "training_minimum": float(minimum[position]),
                "training_maximum": float(maximum[position]),
                "active": bool(active[position]),
                "inactive_reason": (
                    "" if active[position] else "ZERO_VARIANCE_VALID_OUTER_TRAINING"
                ),
            }
        )
    return active, pd.DataFrame(records)


__all__ = [
    "AGGREGATE_STATE",
    "BINARY_FIRM",
    "CATEGORICAL_FIRM",
    "CATEGORICAL_MULTIHOT_FIRM",
    "CONTINUOUS_FIRM",
    "EXPECTED_CATEGORY_COLUMN_COUNT",
    "EXPECTED_ECONOMIC_FEATURE_COUNT",
    "EXPECTED_ELASTIC_NET_INTERACTION_COUNT",
    "EXPECTED_FROZEN_COLUMN_COUNT",
    "CrossSectionRankResult",
    "PastOnlyZScoreResult",
    "SelectorColumn",
    "SelectorMatrixSchema",
    "SelectorPreprocessingResult",
    "build_elastic_net_interaction_registry",
    "build_selector_matrix",
    "build_selector_schema",
    "past_only_expanding_zscore",
    "winsorized_centered_rank",
]
