"""Closed code identity shared by Stage 11 build and verification paths.

This module deliberately has no dependency on the empirical stream,
candidate validator, independent verifier, or predictor adapter.  Those
modules can therefore capture and reauthenticate the exact same code map
without creating an import cycle.
"""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType
from typing import Mapping

from portfolio_bot.validation.validation_data_contract import (
    PROJECT_ROOT,
    canonical_json_bytes,
    sha256_file,
)


MATERIALIZATION_CODE_PATHS = (
    "scripts/build_stage_11_validation_data.py",
    "src/portfolio_bot/validation/validation_data_contract.py",
    "src/portfolio_bot/validation/validation_data_candidate_builder.py",
    "src/portfolio_bot/validation/validation_feature_input_adapter.py",
    "src/portfolio_bot/validation/validation_feature_semantics.py",
    "src/portfolio_bot/validation/validation_empirical_feature_stream.py",
    "src/portfolio_bot/validation/validation_fundamental_adapter.py",
    "src/portfolio_bot/validation/validation_development_parity.py",
    "src/portfolio_bot/validation/validation_universe_auxiliary.py",
    "src/portfolio_bot/validation/validation_build_code_identity.py",
    "src/portfolio_bot/validation/validation_data_materialization.py",
    "src/portfolio_bot/validation/validation_data_independent_verification.py",
    "src/portfolio_bot/validation/predictor_adapter.py",
    "src/portfolio_bot/features/feature_lineage.py",
    "src/portfolio_bot/features/feature_math.py",
    "src/portfolio_bot/features/feature_store_contract.py",
    "src/portfolio_bot/features/feature_store_inputs.py",
    "src/portfolio_bot/features/feature_store_materialization.py",
    "src/portfolio_bot/features/crsp_features.py",
    "src/portfolio_bot/features/fast_crsp_features.py",
    "src/portfolio_bot/features/crsp_cross_section.py",
    "src/portfolio_bot/features/materialize_crsp.py",
    "src/portfolio_bot/features/materialize_crsp_streaming.py",
    "src/portfolio_bot/features/materialize_crsp_routing.py",
    "src/portfolio_bot/features/fundamental_features.py",
    "src/portfolio_bot/features/materialize_fundamentals.py",
    "src/portfolio_bot/features/ownership_event_features.py",
    "src/portfolio_bot/features/materialize_ownership_events.py",
    "src/portfolio_bot/features/optionmetrics_features.py",
    "src/portfolio_bot/features/materialize_optionmetrics.py",
    "src/portfolio_bot/features/aggregate_state_features.py",
    "src/portfolio_bot/features/materialize_aggregate_state.py",
    "src/portfolio_bot/risk_cost/risk_math.py",
    "src/portfolio_bot/risk_cost/cost_math.py",
    "src/portfolio_bot/risk_cost/materialization.py",
)


class ValidationBuildCodeIdentityError(RuntimeError):
    """Raised when the closed Stage 11 implementation set cannot be hashed."""


def materialization_code_hashes(
    *, project_root: Path = PROJECT_ROOT
) -> Mapping[str, str]:
    """Hash the exact closed Stage 11/frozen-formula implementation set."""

    root = project_root.resolve()
    observed: dict[str, str] = {}
    for relative in MATERIALIZATION_CODE_PATHS:
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise ValidationBuildCodeIdentityError(
                f"Missing or unsafe materialization code binding: {relative}"
            )
        observed[relative] = sha256_file(path)
    return MappingProxyType(dict(sorted(observed.items())))


def materialization_code_hashes_sha256(hashes: Mapping[str, str]) -> str:
    """Return the canonical digest of one exact materialization code map."""

    import hashlib

    return hashlib.sha256(canonical_json_bytes(dict(hashes))).hexdigest()


__all__ = [
    "MATERIALIZATION_CODE_PATHS",
    "ValidationBuildCodeIdentityError",
    "materialization_code_hashes",
    "materialization_code_hashes_sha256",
]
