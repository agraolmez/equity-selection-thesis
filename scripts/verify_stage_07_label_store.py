"""Run independent Stage 7 label-store verification."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence


# Set these before importing NumPy/PyArrow through either verification module.
# The numerical implementation is explicitly single-threaded and the frozen
# build policy requires deterministic thread count one.
_DETERMINISTIC_THREAD_ENV = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
)
for _name in _DETERMINISTIC_THREAD_ENV:
    os.environ[_name] = "1"
os.environ["OMP_DYNAMIC"] = "FALSE"

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from portfolio_bot.features.stage7_independent_recompute import (  # noqa: E402
    IndependentRecomputeError,
    recompute_stage7_labels,
)
from portfolio_bot.features.verify_label_store import (  # noqa: E402
    DEFAULT_POLICY,
    Stage7VerificationError,
    verify_stage7_label_store,
)


class CliArgumentError(ValueError):
    """Raised instead of emitting argparse's non-JSON error text."""


class _JsonArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise CliArgumentError(message)


def _parser() -> argparse.ArgumentParser:
    parser = _JsonArgumentParser(
        description="Independently verify the frozen Stage 7 label store."
    )
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--database-root", type=Path)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--primary-pointer", type=Path)
    parser.add_argument("--primary-manifest", type=Path)
    parser.add_argument(
        "--batch-rows",
        type=int,
        default=65_536,
        help="Rows per bounded Arrow/Parquet scan batch (default: 65536).",
    )
    parser.add_argument(
        "--no-publish",
        action="store_true",
        help="Run every check without publishing Stage 8 authority.",
    )
    return parser


def _path(value: Path | None) -> str | None:
    return None if value is None else value.as_posix()


def _json_line(payload: object) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def main(argv: Sequence[str] | None = None) -> int:
    try:
        arguments = _parser().parse_args(argv)
    except CliArgumentError as exc:
        print(
            _json_line(
                {
                    "status": "verification_error",
                    "error": str(exc),
                    "error_type": "CliArgumentError",
                }
            ),
            file=sys.stderr,
        )
        return 2
    if isinstance(arguments.batch_rows, bool) or arguments.batch_rows <= 0:
        print(
            _json_line(
                {
                    "status": "verification_error",
                    "error": "Stage 7 verification batch rows must be positive",
                }
            ),
            file=sys.stderr,
        )
        return 2

    def recompute_hook(artifacts: object):
        return recompute_stage7_labels(
            artifacts,
            batch_rows=arguments.batch_rows,
        )

    try:
        result = verify_stage7_label_store(
            recompute_hook=recompute_hook,
            policy_path=arguments.policy,
            database_root=arguments.database_root,
            artifact_root=arguments.artifact_root,
            primary_pointer_path=arguments.primary_pointer,
            primary_manifest_path=arguments.primary_manifest,
            publish=not arguments.no_publish,
        )
    except (Stage7VerificationError, IndependentRecomputeError) as exc:
        print(
            _json_line(
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
        "differences": [item.as_dict() for item in result.differences],
        "verification_path": _path(result.verification_path),
        "verified_pointer_path": _path(result.verified_pointer_path),
        "published": result.verified_pointer_path is not None,
    }
    print(_json_line(payload))
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
