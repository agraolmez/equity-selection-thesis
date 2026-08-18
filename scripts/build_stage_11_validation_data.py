#!/usr/bin/env python3
"""Compose verified Stage 11 predictor histories and build candidate tables."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from portfolio_bot.validation.validation_data_contract import (  # noqa: E402
    validate_validation_data_contract,
)
from portfolio_bot.validation.validation_empirical_feature_stream import (  # noqa: E402
    load_and_reauthenticate_empirical_candidate_build,
    write_empirical_validation_candidate,
)
from portfolio_bot.validation.validation_feature_input_adapter import (  # noqa: E402
    compose_verified_validation_feature_inputs,
    load_validation_feature_input_bundle,
)
from portfolio_bot.validation.validation_data_independent_verification import (  # noqa: E402
    independently_verify_candidate_manifest,
    write_independent_verification_report,
)
from portfolio_bot.validation.validation_data_materialization import (  # noqa: E402
    validate_candidate_bundle,
    write_candidate_manifest,
)
from portfolio_bot.validation.validation_development_parity import (  # noqa: E402
    load_and_reauthenticate_development_parity,
    prove_development_code_identity_parity,
    write_development_parity_receipt,
)
from portfolio_bot.validation.predictor_adapter import (  # noqa: E402
    write_verified_validation_predictor_manifest,
)


def _path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def _progress(event: str, details: Mapping[str, Any]) -> None:
    print(
        json.dumps(
            {"event": event, "details": dict(details)},
            sort_keys=True,
            separators=(",", ":"),
        ),
        flush=True,
    )


def _add_compose_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--validation-universe-root", required=True, type=_path)
    parser.add_argument("--validation-auxiliary-root", required=True, type=_path)
    parser.add_argument("--full-predictor-context-root", required=True, type=_path)
    parser.add_argument("--feature-input-root", required=True, type=_path)


def _add_build_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--feature-input-root", required=True, type=_path)
    parser.add_argument("--candidate-root", required=True, type=_path)
    parser.add_argument("--crsp-spool-root", required=True, type=_path)
    parser.add_argument("--empirical-build-receipt", required=True, type=_path)
    parser.add_argument("--batch-size", type=int, default=2_048)
    parser.add_argument("--crsp-checkpoint-rows", type=int, default=4_096)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=(
            "Build only the 2014-11 context and 2014-12--2019-10 validation "
            "predictors/risk inputs. Labels, outcomes, weights, trades, P&L, "
            "and 2020+ data remain inaccessible."
        )
    )
    commands = result.add_subparsers(dest="command", required=True)
    compose = commands.add_parser("compose-feature-inputs")
    _add_compose_paths(compose)
    build = commands.add_parser("build-candidate")
    _add_build_paths(build)
    both = commands.add_parser("compose-and-build")
    _add_compose_paths(both)
    both.add_argument("--candidate-root", required=True, type=_path)
    both.add_argument("--crsp-spool-root", required=True, type=_path)
    both.add_argument("--empirical-build-receipt", required=True, type=_path)
    both.add_argument("--batch-size", type=int, default=2_048)
    both.add_argument("--crsp-checkpoint-rows", type=int, default=4_096)

    parity = commands.add_parser(
        "prove-development-parity",
        help=(
            "Rehash the complete verified Stage 6 input/output identity and "
            "prove the generalized development scope is an exact no-op."
        ),
    )
    parity.add_argument("--parity-receipt", required=True, type=_path)

    finalize = commands.add_parser(
        "finalize-candidate",
        help=(
            "Reauthenticate development parity, fully validate the 12-table "
            "candidate, independently verify it, and write the predictor adapter manifest."
        ),
    )
    finalize.add_argument("--candidate-root", required=True, type=_path)
    finalize.add_argument("--parity-receipt", required=True, type=_path)
    finalize.add_argument("--empirical-build-receipt", required=True, type=_path)
    finalize.add_argument("--candidate-manifest", required=True, type=_path)
    finalize.add_argument("--verification-report", required=True, type=_path)
    finalize.add_argument("--predictor-manifest", required=True, type=_path)

    all_in_one = commands.add_parser(
        "prove-and-finalize-candidate",
        help="Create byte-authenticated parity evidence and finalize one candidate.",
    )
    all_in_one.add_argument("--candidate-root", required=True, type=_path)
    all_in_one.add_argument("--parity-receipt", required=True, type=_path)
    all_in_one.add_argument("--empirical-build-receipt", required=True, type=_path)
    all_in_one.add_argument("--candidate-manifest", required=True, type=_path)
    all_in_one.add_argument("--verification-report", required=True, type=_path)
    all_in_one.add_argument("--predictor-manifest", required=True, type=_path)
    return result


def _compose(arguments: argparse.Namespace, authority):
    return compose_verified_validation_feature_inputs(
        predictor_authority=authority,
        validation_universe_root=arguments.validation_universe_root,
        validation_auxiliary_root=arguments.validation_auxiliary_root,
        full_predictor_context_root=arguments.full_predictor_context_root,
        output_root=arguments.feature_input_root,
        progress=_progress,
    )


def _build(arguments: argparse.Namespace, authority, bundle):
    return write_empirical_validation_candidate(
        bundle=bundle,
        destination=arguments.candidate_root,
        crsp_spool_path=arguments.crsp_spool_root,
        empirical_build_receipt_path=arguments.empirical_build_receipt,
        contract=authority,
        batch_size=arguments.batch_size,
        crsp_checkpoint_rows=arguments.crsp_checkpoint_rows,
        progress=_progress,
    )


def _finalize(arguments: argparse.Namespace, authority, parity_authority):
    empirical_build = load_and_reauthenticate_empirical_candidate_build(
        arguments.empirical_build_receipt,
        authority,
    )
    primary = validate_candidate_bundle(
        arguments.candidate_root,
        development_parity=parity_authority,
        empirical_build=empirical_build,
    )
    manifest_path = write_candidate_manifest(
        primary,
        arguments.candidate_manifest,
    )
    independent = independently_verify_candidate_manifest(
        manifest_path,
        arguments.candidate_root,
    )
    report_path = write_independent_verification_report(
        independent,
        arguments.verification_report,
    )
    predictor = write_verified_validation_predictor_manifest(
        candidate_manifest_path=manifest_path,
        verification_report_path=report_path,
        candidate_root=arguments.candidate_root,
        destination=arguments.predictor_manifest,
    )
    return {
        "event": "stage11_validation_candidate_independently_verified",
        "build_id": independent.build_id,
        "verification_id": independent.verification_id,
        "candidate_manifest": manifest_path.as_posix(),
        "candidate_manifest_sha256": independent.manifest_sha256,
        "empirical_build_id": empirical_build.receipt.build_id,
        "empirical_build_receipt": empirical_build.receipt_path.as_posix(),
        "empirical_build_receipt_sha256": empirical_build.receipt_sha256,
        "verification_report": report_path.as_posix(),
        "predictor_manifest": predictor.manifest_path.as_posix(),
        "predictor_manifest_sha256": predictor.manifest_sha256,
        "candidate_feature_rows": primary.feature_audit.rows,
        "candidate_lineage_rows": sum(
            value.rows for value in primary.lineage_audits.values()
        ),
        "development_parity_method": parity_authority.receipt.method,
        "development_empirical_replay_feature_rows": (
            parity_authority.receipt.empirical_replay_feature_rows
        ),
        "development_empirical_replay_lineage_rows": (
            parity_authority.receipt.empirical_replay_lineage_rows
        ),
        "development_empirical_difference_count": (
            parity_authority.receipt.empirical_difference_count
        ),
        "development_reference_feature_rows_authenticated": (
            parity_authority.receipt.feature_rows
        ),
        "development_reference_lineage_rows_authenticated": (
            parity_authority.receipt.lineage_rows
        ),
        "validation_outcome_rows_read": 0,
        "final_period_rows_read": 0,
        "outcome_unblind_used": False,
    }


def main() -> int:
    arguments = parser().parse_args()
    # This is the no-outcome predictor technical authority.  It is deliberately
    # established before any caller-supplied data path is inspected.
    authority = validate_validation_data_contract()
    if arguments.command == "prove-development-parity":
        parity_authority = prove_development_code_identity_parity(authority)
        receipt_path = write_development_parity_receipt(
            parity_authority,
            arguments.parity_receipt,
        )
        output = {
            "event": "stage11_development_code_identity_parity_complete",
            "parity_receipt": receipt_path.as_posix(),
            "status": parity_authority.receipt.status,
            "method": parity_authority.receipt.method,
            "reference_feature_rows_authenticated": (
                parity_authority.receipt.feature_rows
            ),
            "reference_lineage_rows_authenticated": (
                parity_authority.receipt.lineage_rows
            ),
            "empirical_replay_feature_rows": (
                parity_authority.receipt.empirical_replay_feature_rows
            ),
            "empirical_replay_lineage_rows": (
                parity_authority.receipt.empirical_replay_lineage_rows
            ),
            "identity_mismatch_count": 0,
            "empirical_difference_count": (
                parity_authority.receipt.empirical_difference_count
            ),
            "evidence_sha256": parity_authority.receipt.evidence_sha256,
            "outcome_unblind_used": False,
        }
    elif arguments.command == "finalize-candidate":
        parity_authority = load_and_reauthenticate_development_parity(
            arguments.parity_receipt,
            authority,
        )
        output = _finalize(arguments, authority, parity_authority)
    elif arguments.command == "prove-and-finalize-candidate":
        parity_authority = prove_development_code_identity_parity(authority)
        receipt_path = write_development_parity_receipt(
            parity_authority,
            arguments.parity_receipt,
        )
        output = {
            **_finalize(arguments, authority, parity_authority),
            "parity_receipt": receipt_path.as_posix(),
        }
    elif arguments.command == "compose-feature-inputs":
        bundle = _compose(arguments, authority)
        output = {
            "event": "stage11_feature_input_bundle_complete",
            "receipt_path": bundle.receipt_path.as_posix(),
            "input_bundle_sha256": bundle.input_bundle_sha256,
            "candidate_materialized": False,
            "outcome_unblind_used": False,
        }
    elif arguments.command == "build-candidate":
        bundle = load_validation_feature_input_bundle(
            arguments.feature_input_root,
            predictor_authority=authority,
        )
        built = _build(arguments, authority, bundle)
        output = {
            "event": "stage11_validation_candidate_complete",
            "candidate_root": built.feature_receipt.destination.as_posix(),
            "feature_rows": built.feature_receipt.feature_rows,
            "lineage_rows": built.feature_receipt.lineage_rows,
            "scored_build_id": built.scored_build_id,
            "optionmetrics_working_build_id": built.optionmetrics_working_build_id,
            "input_bundle_sha256": built.input_bundle_sha256,
            "crsp_router_reused": built.crsp_router_reused,
            "empirical_build_id": built.empirical_build.receipt.build_id,
            "empirical_build_receipt": built.empirical_build.receipt_path.as_posix(),
            "empirical_build_receipt_sha256": built.empirical_build.receipt_sha256,
            "outcome_unblind_used": False,
        }
    elif arguments.command == "compose-and-build":
        bundle = _compose(arguments, authority)
        built = _build(arguments, authority, bundle)
        output = {
            "event": "stage11_feature_inputs_and_candidate_complete",
            "input_receipt_path": bundle.receipt_path.as_posix(),
            "candidate_root": built.feature_receipt.destination.as_posix(),
            "feature_rows": built.feature_receipt.feature_rows,
            "lineage_rows": built.feature_receipt.lineage_rows,
            "input_bundle_sha256": built.input_bundle_sha256,
            "crsp_router_reused": built.crsp_router_reused,
            "empirical_build_id": built.empirical_build.receipt.build_id,
            "empirical_build_receipt": built.empirical_build.receipt_path.as_posix(),
            "empirical_build_receipt_sha256": built.empirical_build.receipt_sha256,
            "outcome_unblind_used": False,
        }
    else:  # pragma: no cover - argparse keeps the command set closed.
        raise RuntimeError(f"Unhandled command: {arguments.command}")
    print(json.dumps(output, sort_keys=True, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
