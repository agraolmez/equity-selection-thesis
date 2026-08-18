#!/usr/bin/env python3
"""Build isolated Stage 11 universe/auxiliary predictor inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from portfolio_bot.validation.validation_universe_auxiliary import (
    FullPredictorContextInputs,
    ValidationUniverseInputs,
    build_validation_auxiliary_normalization,
    compose_full_validation_predictor_context,
    compose_validation_universe_inputs,
    compute_validation_universe,
    write_validation_universe,
)


def _progress(event: str, details: Mapping[str, Any]) -> None:
    print(json.dumps({"event": event, "details": dict(details)}, sort_keys=True), flush=True)


def _path(value: str) -> Path:
    return Path(value).expanduser().resolve()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    commands = result.add_subparsers(dest="command", required=True)

    compose = commands.add_parser("compose-universe-context")
    compose.add_argument("--development-typed-root", required=True, type=_path)
    compose.add_argument("--development-return-root", required=True, type=_path)
    compose.add_argument("--validation-market-root", required=True, type=_path)
    compose.add_argument("--validation-return-root", required=True, type=_path)
    compose.add_argument("--calendar", required=True, type=_path)
    compose.add_argument("--output-root", required=True, type=_path)

    universe = commands.add_parser("build-universe")
    universe.add_argument("--typed-root", required=True, type=_path)
    universe.add_argument("--return-root", required=True, type=_path)
    universe.add_argument("--calendar", required=True, type=_path)
    universe.add_argument("--output-root", required=True, type=_path)

    auxiliary = commands.add_parser("build-auxiliary")
    auxiliary.add_argument("--raw-root", required=True, type=_path)
    auxiliary.add_argument("--development-auxiliary-root", required=True, type=_path)
    auxiliary.add_argument("--calendar", required=True, type=_path)
    auxiliary.add_argument("--validation-identity", required=True, type=_path)
    auxiliary.add_argument("--scored-universe", required=True, type=_path)
    auxiliary.add_argument("--internal-context-universe", required=True, type=_path)
    auxiliary.add_argument("--output-root", required=True, type=_path)

    full = commands.add_parser("compose-full-predictor-context")
    full.add_argument("--development-typed-root", required=True, type=_path)
    full.add_argument("--development-return-root", required=True, type=_path)
    full.add_argument("--development-auxiliary-root", required=True, type=_path)
    full.add_argument("--validation-market-root", required=True, type=_path)
    full.add_argument("--validation-return-root", required=True, type=_path)
    full.add_argument("--validation-auxiliary-root", required=True, type=_path)
    full.add_argument("--calendar", required=True, type=_path)
    full.add_argument("--output-root", required=True, type=_path)
    return result


def main() -> None:
    arguments = parser().parse_args()
    if arguments.command == "compose-universe-context":
        _, receipt = compose_validation_universe_inputs(
            development_typed_root=arguments.development_typed_root,
            development_return_root=arguments.development_return_root,
            validation_market_root=arguments.validation_market_root,
            validation_return_root=arguments.validation_return_root,
            calendar_path=arguments.calendar,
            output_root=arguments.output_root,
            progress=_progress,
        )
    elif arguments.command == "build-universe":
        computed = compute_validation_universe(
            ValidationUniverseInputs(
                arguments.typed_root,
                arguments.return_root,
                arguments.calendar,
            )
        )
        receipt = write_validation_universe(computed, arguments.output_root)
    elif arguments.command == "build-auxiliary":
        receipt = build_validation_auxiliary_normalization(
            raw_root=arguments.raw_root,
            development_auxiliary_root=arguments.development_auxiliary_root,
            trading_calendar_path=arguments.calendar,
            validation_identity_path=arguments.validation_identity,
            scored_universe_path=arguments.scored_universe,
            internal_context_universe_path=arguments.internal_context_universe,
            output_root=arguments.output_root,
            progress=_progress,
        )
    else:
        receipt = compose_full_validation_predictor_context(
            FullPredictorContextInputs(
                development_typed_root=arguments.development_typed_root,
                development_return_root=arguments.development_return_root,
                development_auxiliary_root=arguments.development_auxiliary_root,
                validation_market_root=arguments.validation_market_root,
                validation_return_root=arguments.validation_return_root,
                validation_auxiliary_root=arguments.validation_auxiliary_root,
                calendar_path=arguments.calendar,
            ),
            arguments.output_root,
            progress=_progress,
        )
    print(json.dumps(receipt, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
