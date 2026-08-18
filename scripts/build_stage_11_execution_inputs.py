#!/usr/bin/env python3
"""No-argument, no-outcome Stage 11 execution-input publisher."""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from portfolio_bot.validation.execution_input_builder import (  # noqa: E402
    Stage11ExecutionInputBuildError,
    build_stage11_execution_inputs,
)


def main(argv: list[str] | None = None) -> int:
    selected = sys.argv[1:] if argv is None else argv
    if selected:
        print(
            json.dumps(
                {
                    "status": "execution_input_build_sealed",
                    "error": "caller-supplied paths and options are prohibited",
                    "validation_outcome_rows_read": 0,
                    "final_2020_2025_rows_read": 0,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    try:
        result = build_stage11_execution_inputs()
        print(
            json.dumps(
                {
                    "status": "STAGE11_EXECUTION_INPUTS_BUILT_PENDING_INDEPENDENT_VERIFICATION",
                    "calendar_path": str(result.calendar_path),
                    "calendar_sha256": result.calendar_sha256,
                    "calendar_rows": result.calendar_rows,
                    "outcome_manifest_sha256": result.outcome_manifest_sha256,
                    "execution_manifest_sha256": result.execution_manifest_sha256,
                    "execution_manifest_payload_sha256": result.execution_manifest_payload_sha256,
                    "gate_b_execution_authorized": False,
                },
                sort_keys=True,
                indent=2,
            )
        )
        return 0
    except (Stage11ExecutionInputBuildError, OSError, RuntimeError, ValueError) as exc:
        print(
            json.dumps(
                {
                    "status": "execution_input_build_sealed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "validation_outcome_rows_read": 0,
                    "forecasts_weights_trades_pnl_rows_created": 0,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
