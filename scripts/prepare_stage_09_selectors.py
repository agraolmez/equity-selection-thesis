#!/usr/bin/env python3
"""Authenticate and prepare Stage 9 selector matrices; never fit a model."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from portfolio_bot.selectors.contract import DEFAULT_POLICY, validate_contract
from portfolio_bot.selectors.inputs import resolve_stage9_inputs
from portfolio_bot.selectors.materialization import (
    compute_prepared_build_identity,
    prepare_stage9_selector_matrices,
    prepared_cache_root,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the four development-only Stage 9 selector matrices. "
            "This command cannot fit models or run a portfolio/backtest."
        )
    )
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument(
        "--identity-only",
        action="store_true",
        help="authenticate inputs and print the deterministic cache identity without writing matrices",
    )
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    policy_path = args.policy.expanduser().resolve()
    validate_contract(policy_path)
    started = time.monotonic()
    print("Stage 9 preparation: authenticating the closed Stage 6--8 inputs...", flush=True)
    inputs = resolve_stage9_inputs(policy_path)
    identity = compute_prepared_build_identity(inputs, policy_path=policy_path)
    root = prepared_cache_root(inputs, identity, policy_path=policy_path)
    if args.identity_only:
        print(
            json.dumps(
                {
                    "build_id": identity.build_id,
                    "content_fingerprint": identity.content_fingerprint,
                    "prepared_cache_root": root.as_posix(),
                    "matrices_written": False,
                    "model_fits_performed": False,
                },
                sort_keys=True,
                indent=2,
            ),
            flush=True,
        )
        return 0

    last_update = 0.0

    def progress(message: str) -> None:
        nonlocal last_update
        now = time.monotonic()
        if now - last_update >= 10.0 or message.startswith("complete"):
            print(
                f"Stage 9 preparation: {message}; elapsed={(now - started) / 60.0:.1f}m",
                flush=True,
            )
            last_update = now

    result = prepare_stage9_selector_matrices(
        inputs=inputs,
        policy_path=policy_path,
        progress=progress,
    )
    print(
        json.dumps(
            {
                "build_id": result.identity.build_id,
                "content_fingerprint": result.identity.content_fingerprint,
                "prepared_cache_root": result.root.as_posix(),
                "status": result.metadata["status"],
                "resumed_complete_build": result.resumed_complete_build,
                "inner_base_matrix_shape": result.metadata["inner_base_matrix_shape"],
                "full_base_matrix_shape": result.metadata["full_base_matrix_shape"],
                "elastic_full_matrix_shape": result.metadata["elastic_full_matrix_shape"],
                "model_fits_performed": False,
                "elapsed_minutes": round((time.monotonic() - started) / 60.0, 3),
            },
            sort_keys=True,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
