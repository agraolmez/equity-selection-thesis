#!/usr/bin/env python3
"""Run independent Stage 8 risk/cost verification."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Sequence


for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
):
    os.environ[_name] = "1"
os.environ["OMP_DYNAMIC"] = "FALSE"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from portfolio_bot.risk_cost.contract import DEFAULT_POLICY, RiskCostContractError  # noqa: E402
from portfolio_bot.risk_cost.independent_verification import (  # noqa: E402
    Stage8VerificationError,
    verify_stage8_risk_cost,
)


class CliArgumentError(ValueError):
    """Raised instead of argparse's non-JSON error text."""


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise CliArgumentError(message)


def _parser() -> argparse.ArgumentParser:
    parser = _Parser(description="Independently verify frozen Stage 8 risk/cost outputs.")
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--primary-pointer", type=Path)
    parser.add_argument("--batch-rows", type=int, default=262_144)
    parser.add_argument(
        "--no-publish",
        action="store_true",
        help="Run every check without replacing Stage 9 authority.",
    )
    return parser


def _line(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def main(argv: Sequence[str] | None = None) -> int:
    try:
        arguments = _parser().parse_args(argv)
    except CliArgumentError as exc:
        print(_line({"status": "verification_error", "error": str(exc)}), file=sys.stderr)
        return 2
    if isinstance(arguments.batch_rows, bool) or arguments.batch_rows <= 0:
        print(
            _line({"status": "verification_error", "error": "batch rows must be positive"}),
            file=sys.stderr,
        )
        return 2
    started = time.monotonic()
    last_update = 0.0

    def progress(message: str) -> None:
        nonlocal last_update
        now = time.monotonic()
        if now - last_update >= 10.0 or message.startswith("immutable"):
            print(
                _line(
                    {
                        "status": "verification_progress",
                        "message": message,
                        "elapsed_minutes": round((now - started) / 60.0, 3),
                    }
                ),
                flush=True,
            )
            last_update = now

    try:
        result = verify_stage8_risk_cost(
            policy_path=arguments.policy,
            primary_pointer_path=arguments.primary_pointer,
            batch_rows=arguments.batch_rows,
            publish=not arguments.no_publish,
            progress=progress,
        )
    except (Stage8VerificationError, RiskCostContractError) as exc:
        print(
            _line(
                {
                    "status": "verification_error",
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                }
            ),
            file=sys.stderr,
        )
        return 2
    payload = {
        "status": "verified_passed" if result.passed else "verified_failed",
        "build_id": result.build_id,
        "verification_id": result.verification_id,
        "difference_count": len(result.differences),
        "differences": [value.as_dict() for value in result.differences],
        "verification_path": (
            None if result.verification_path is None else result.verification_path.as_posix()
        ),
        "verified_pointer_path": (
            None
            if result.verified_pointer_path is None
            else result.verified_pointer_path.as_posix()
        ),
        "published": result.verified_pointer_path is not None,
        "elapsed_minutes": round((time.monotonic() - started) / 60.0, 3),
    }
    print(_line(payload))
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
