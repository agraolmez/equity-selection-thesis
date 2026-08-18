#!/usr/bin/env python3
"""No-argument Stage 11 validation entrypoint behind frozen authority.

Caller-supplied receipts, paths, model settings, and date ranges are prohibited.
The command remains sealed until the distinct execution approval and input
manifest digests are frozen by reviewed repository changes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from portfolio_bot.validation.validation_runner import (  # noqa: E402
    Stage11ValidationSealedError,
    open_stage11_validation_unblind,
)
from stage_11_portfolio_resume import (  # noqa: E402
    run_stage11_authenticated_portfolio_resume,
)


def _progress(message: str) -> None:
    print(
        json.dumps({"status": "running", "message": message}, sort_keys=True),
        file=sys.stderr,
        flush=True,
    )


def main(argv: list[str] | None = None) -> int:
    # There are deliberately no authority-path options.  This also rejects
    # legacy attempts to inject a temporary receipt or pointer.
    selected_argv = sys.argv[1:] if argv is None else argv
    if selected_argv:
        print(
            json.dumps(
                {
                    "status": "validation_sealed",
                    "error_type": "Stage11ValidationSealedError",
                    "error": (
                        "caller-supplied arguments are prohibited; Stage 11 "
                        "uses only repository-frozen authority paths"
                    ),
                    "outcome_rows_read": 0,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    try:
        with open_stage11_validation_unblind() as authority:
            result = run_stage11_authenticated_portfolio_resume(
                authority,
                progress=_progress,
            )
    except Stage11ValidationSealedError as exc:
        print(
            json.dumps(
                {
                    "status": "validation_sealed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "outcome_rows_read": 0,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "validation_failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1
    print(
        json.dumps(
            {
                "status": "validation_execution_verified",
                "build_id": result.publication.build_id,
                "verification_id": result.verification.verification_id,
                "difference_count": result.verification.difference_count,
                "primary_bundle": result.publication.bundle_path.as_posix(),
                "verification_bundle": result.verification.bundle_path.as_posix(),
                "selection_sha256": result.selector_stage.selection.selection_sha256,
                "verified_results_pointer": result.verified_report.pointer_path.as_posix(),
                "results_report_bundle": result.verified_report.bundle_path.as_posix(),
                "results_report_id": result.verified_report.report_id,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
