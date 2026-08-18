#!/usr/bin/env python3
"""Sealed, no-argument independent Stage 11 Gate-A source verifier."""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from portfolio_bot.validation.validation_source_materialization import (  # noqa: E402
    Stage11SourceMaterializationSealedError,
    open_stage11_validation_source_materialization,
)
from portfolio_bot.validation.validation_source_verification import (  # noqa: E402
    Stage11SourceVerificationError,
    verify_stage11_normalized_sources,
)


def main(argv: list[str] | None = None) -> int:
    selected = sys.argv[1:] if argv is None else argv
    if selected:
        print(
            json.dumps(
                {
                    "status": "source_verification_sealed",
                    "error": "caller-supplied paths and options are prohibited",
                    "validation_or_reserve_rows_read": 0,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    try:
        with open_stage11_validation_source_materialization() as authority:
            result = verify_stage11_normalized_sources(authority)
            print(
                json.dumps(
                    {
                        "status": (
                            "STAGE11_SOURCE_MATERIALIZATION_VERIFIED"
                            if result.difference_count == 0
                            else "STAGE11_SOURCE_MATERIALIZATION_VERIFICATION_FAILED"
                        ),
                        "verification_id": result.verification_id,
                        "bundle_path": str(result.bundle_path),
                        "report_sha256": result.report_sha256,
                        "difference_count": result.difference_count,
                        "verified_pointer_sha256": result.verified_pointer_sha256,
                        "gate_b_execution_authorized": False,
                    },
                    sort_keys=True,
                    indent=2,
                )
            )
            return 0 if result.difference_count == 0 else 3
    except (
        Stage11SourceMaterializationSealedError,
        Stage11SourceVerificationError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(
            json.dumps(
                {
                    "status": "source_verification_sealed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "labels_forecasts_weights_trades_pnl_rows_created": 0,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
