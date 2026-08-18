#!/usr/bin/env python3
"""No-argument, performance-blind Stage 11 portfolio diagnostic entrypoint."""

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
from portfolio_bot.validation.stage11_diagnostics import (  # noqa: E402
    Stage11DiagnosticPersistenceError,
)
from stage_11_portfolio_resume import (  # noqa: E402
    run_stage11_authenticated_diagnostics,
)


def _emit(value: dict[str, object], *, error: bool = False) -> None:
    print(
        json.dumps(value, sort_keys=True),
        file=sys.stderr if error else sys.stdout,
        flush=True,
    )


def _progress(message: str) -> None:
    _emit(
        {
            "status": "diagnostic_running",
            "message": message,
            "performance_fields_emitted": 0,
        },
        error=True,
    )


def main(argv: list[str] | None = None) -> int:
    selected_argv = sys.argv[1:] if argv is None else argv
    if selected_argv:
        _emit(
            {
                "status": "diagnostic_sealed",
                "error_type": "Stage11ValidationSealedError",
                "error_code": "caller_arguments_prohibited",
                "performance_fields_emitted": 0,
            },
            error=True,
        )
        return 2
    try:
        with open_stage11_validation_unblind() as authority:
            result = run_stage11_authenticated_diagnostics(
                authority,
                progress=_progress,
            )
    except Stage11ValidationSealedError as exc:
        private_cause = exc.__cause__
        if isinstance(private_cause, Stage11DiagnosticPersistenceError):
            _emit(
                {
                    "status": "diagnostic_infrastructure_failed",
                    "error_type": type(private_cause).__name__,
                    "error_code": "diagnostic_persistence_failed",
                    "performance_fields_emitted": 0,
                },
                error=True,
            )
            return 1
        _emit(
            {
                "status": "diagnostic_sealed",
                "error_type": "Stage11ValidationSealedError",
                "error_code": "authenticated_unblind_authority_unavailable",
                "private_cause_type": (
                    None
                    if private_cause is None
                    else type(private_cause).__name__
                ),
                "performance_fields_emitted": 0,
            },
            error=True,
        )
        return 2
    except Exception as exc:
        _emit(
            {
                "status": "diagnostic_infrastructure_failed",
                "error_type": type(exc).__name__,
                "error_code": "inspect_private_diagnostic_evidence",
                "performance_fields_emitted": 0,
            },
            error=True,
        )
        return 1

    status = result.portfolio_status
    if not result.complete:
        _emit(
            {
                "status": "diagnostic_month_failed",
                "completed_month_count": status.completed_month_count,
                "next_holding_month": status.next_holding_month,
                "failure_report_sha256": status.failure_report_sha256,
                "failure_count": status.failure_count,
                "checkpoint_advanced": False,
                "performance_fields_emitted": 0,
            }
        )
        return 1
    assert result.completion is not None
    _emit(
        {
            "status": "diagnostic_path_complete",
            "completed_month_count": result.completion.completed_month_count,
            "completion_sha256": result.completion.completion_sha256,
            "latest_checkpoint_sha256": (
                result.completion.latest_checkpoint_sha256
            ),
            "regression_suite_sha256": result.regression_suite_sha256,
            "performance_fields_emitted": 0,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
