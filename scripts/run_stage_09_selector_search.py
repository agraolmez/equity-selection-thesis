#!/usr/bin/env python3
"""Run the complete development-only Stage 9 selector search and seal it."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from portfolio_bot.selectors.contract import DEFAULT_POLICY, validate_contract
from portfolio_bot.selectors.inputs import resolve_stage9_inputs
from portfolio_bot.selectors.materialization import (
    compute_prepared_build_identity,
    prepared_cache_root,
)
from portfolio_bot.selectors import search_execution as selector_search_execution
from portfolio_bot.selectors.search_execution import (
    Stage9SearchExecutionError,
    execute_stage9_selector_search,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Execute the frozen Stage 9 development OOF selector search. "
            "The command cannot read 2015+, form portfolios, calculate P&L, "
            "or authorize Stage 10."
        )
    )
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument(
        "--prepared-root",
        type=Path,
        default=None,
        help=(
            "Optional explicit prepared-cache path. Its identity must still "
            "equal the cache derived from the verified Stage 6--8 inputs."
        ),
    )
    parser.add_argument(
        "--heartbeat-seconds",
        type=float,
        default=30.0,
        help="Maximum quiet interval before a worker progress heartbeat.",
    )
    parser.add_argument(
        "--elastic-net-fold1-watchdog-seconds",
        type=float,
        default=1_800.0,
        help=(
            "Operational Elastic Net fold-1 watchdog. This does not change "
            "the model or its numerical acceptance gates."
        ),
    )
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument(
        "--resume-authenticated-elastic-net-cache",
        action="store_true",
        help=(
            "Require and reuse all four already authenticated Elastic Net "
            "parents after a prior cold run completed. No Elastic Net worker "
            "is launched; all cache, convergence, dual-gap, and KKT gates "
            "remain mandatory."
        ),
    )
    resume_group.add_argument(
        "--resume-authenticated-elastic-net-prefix",
        action="store_true",
        help=(
            "Require an authenticated completed Elastic Net fold-1 parent, "
            "then reuse any other completed fold parents and cold-fit only "
            "the missing later folds. The fold-1 runtime deadline remains "
            "active while its parent is reauthenticated."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.monotonic()
    try:
        watchdog_seconds = float(args.elastic_net_fold1_watchdog_seconds)
        if not math.isfinite(watchdog_seconds) or watchdog_seconds <= 0.0:
            raise ValueError("Elastic Net fold-1 watchdog must be finite and positive")
        # The watchdog is an operational controller setting, not a model-fit
        # input. Controller/script changes are excluded from family fit
        # identities, so authenticated fit caches remain reusable; the final
        # search bundle independently binds both controller files.
        selector_search_execution._ELASTIC_NET_FOLD1_RUNTIME_LIMIT_SECONDS = (
            watchdog_seconds
        )
        policy_path = args.policy.expanduser().resolve()
        validate_contract(policy_path)
        print(
            "Stage 9 search: authenticating verified inputs and prepared identity...",
            flush=True,
        )
        inputs = resolve_stage9_inputs(policy_path)
        identity = compute_prepared_build_identity(inputs, policy_path=policy_path)
        canonical_root = prepared_cache_root(
            inputs, identity, policy_path=policy_path
        )
        prepared_root = (
            canonical_root
            if args.prepared_root is None
            else args.prepared_root.expanduser().resolve()
        )
        if prepared_root != canonical_root:
            raise Stage9SearchExecutionError(
                "Explicit prepared root differs from the verified content-addressed cache"
            )

        def progress(message: str) -> None:
            elapsed = (time.monotonic() - started) / 60.0
            print(f"Stage 9 search: {message}; elapsed={elapsed:.1f}m", flush=True)

        result = execute_stage9_selector_search(
            prepared_root=prepared_root,
            expected_build_id=identity.build_id,
            expected_content_fingerprint=identity.content_fingerprint,
            database_root=inputs.database_root,
            policy_path=policy_path,
            progress=progress,
            heartbeat_seconds=args.heartbeat_seconds,
            resume_authenticated_elastic_net_cache=bool(
                args.resume_authenticated_elastic_net_cache
            ),
            resume_authenticated_elastic_net_prefix=bool(
                args.resume_authenticated_elastic_net_prefix
            ),
        )
    except (RuntimeError, ValueError, OSError) as exc:
        print(
            json.dumps(
                {
                    "status": "stage09_search_error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "elapsed_minutes": round(
                        (time.monotonic() - started) / 60.0, 3
                    ),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )
        return 2
    print(
        json.dumps(
            {
                "status": "COMPLETED_SEARCH_BUNDLE_SEALED",
                "search_id": result.publication.search_id,
                "search_bundle": str(result.publication.path),
                "search_receipt_sha256": result.publication.receipt_sha256,
                "sealed_xgboost_core_receipt": str(result.sealed_core.path),
                "sealed_xgboost_core_sha256": result.sealed_core.sha256,
                "gamma_parent_config_ids": list(
                    result.sealed_core.gamma_parent_config_ids
                ),
                "evaluated_configurations": result.evaluated_configurations,
                "fit_receipts": result.fit_receipts,
                "fit_cache_hits": result.cache_hits,
                "reused_completed_search": result.publication.reused_existing,
                "stage_10_authorized": False,
                "stage_11_or_later_authorized": False,
                "portfolio_or_backtest_created": False,
                "validation_or_final_rows_read": 0,
                "elastic_net_fold1_watchdog_seconds": watchdog_seconds,
                "resumed_authenticated_elastic_net_cache": bool(
                    args.resume_authenticated_elastic_net_cache
                ),
                "resumed_authenticated_elastic_net_prefix": bool(
                    args.resume_authenticated_elastic_net_prefix
                ),
                "elapsed_minutes": round(result.elapsed_seconds / 60.0, 3),
            },
            sort_keys=True,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
