#!/usr/bin/env python3
"""Build the Stage 7 development label store and primary audit artifacts."""

from __future__ import annotations

import argparse
import json
import sys
import time
import tomllib
from pathlib import Path

SCRIPT_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_PROJECT_ROOT / "src"))

from portfolio_bot.features.label_return_router import (
    spool_label_return_router,
)
from portfolio_bot.features.label_store_contract import (
    DEFAULT_POLICY,
    PROJECT_ROOT,
)
from portfolio_bot.features.label_store_inputs import (
    resolve_label_input_catalog,
)
from portfolio_bot.features.label_store_materialization import (
    compute_label_store_build_identity,
    load_development_label_context,
    materialize_label_store_cache,
)
from portfolio_bot.features.label_store_publication import (
    publish_primary_label_store,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize development-only Stage 7 labels. This command does "
            "not train models, construct portfolios, or run a backtest."
        )
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=DEFAULT_POLICY,
        help="Hash-bound Stage 7 publication policy.",
    )
    return parser.parse_args()


def _progress(started: float):
    last_reported = -1

    def report(values) -> None:
        nonlocal last_reported
        complete = int(values["months_complete"])
        total = int(values["months_total"])
        if complete != total and complete % 10 and complete == last_reported:
            return
        if complete != total and complete % 10:
            return
        last_reported = complete
        elapsed = time.monotonic() - started
        print(
            "Stage 7 monthly cache: "
            f"{complete}/{total}; "
            f"audit={int(values['audit_rows']):,}; "
            f"long={int(values['long_rows']):,}; "
            f"lineage={int(values['lineage_rows']):,}; "
            f"elapsed={elapsed / 60.0:.1f}m",
            flush=True,
        )

    return report


def main() -> int:
    args = _arguments()
    policy_path = args.policy.expanduser().resolve()
    with policy_path.open("rb") as handle:
        policy = tomllib.load(handle)
    rules_path = Path(policy["physical_rules"]["path"])
    if not rules_path.is_absolute():
        rules_path = PROJECT_ROOT / rules_path
    rules = json.loads(rules_path.read_text("utf-8"))

    started = time.monotonic()
    print("Stage 7.3: authenticating six closed inputs...", flush=True)
    inputs = resolve_label_input_catalog(policy_path)
    identity = compute_label_store_build_identity(
        inputs,
        policy_path=policy_path,
    )
    print(f"Stage 7 build ID: {identity.build_id}", flush=True)

    context = load_development_label_context(
        inputs,
        policy_path=policy_path,
    )
    database_root = inputs.database_root
    cache_root = database_root / policy["database"]["cache_root"]
    router_path = cache_root / "return_router" / identity.build_id
    print(
        "Stage 7.3: routing the bounded return table once "
        f"for {context.long_intervals.num_rows:,} security-months...",
        flush=True,
    )
    router = spool_label_return_router(
        inputs,
        context.long_intervals,
        router_path,
        identity.build_id,
        identity.content_fingerprint,
        row_group_size=int(policy["build"]["row_group_size"]),
        compression=str(policy["build"]["compression"]),
    )
    print(
        "Stage 7 return router: "
        f"{router.routed_row_count:,} rows; "
        f"{'reused' if router.reused_existing else 'built'}",
        flush=True,
    )

    cache = materialize_label_store_cache(
        inputs=inputs,
        identity=identity,
        context=context,
        return_router=router,
        rules=rules,
        cache_root=cache_root,
        progress=_progress(started),
        policy_path=policy_path,
    )
    print(
        "Stage 7.3: publishing authenticated interim, processed, and "
        "quality bundles...",
        flush=True,
    )
    publication = publish_primary_label_store(
        inputs=inputs,
        cache_result=cache,
        rules=rules,
        policy_path=policy_path,
    )
    print(
        json.dumps(
            {
                "build_id": publication.build_id,
                "status": publication.pointer["status"],
                "interim_path": publication.interim_path.as_posix(),
                "processed_path": publication.processed_path.as_posix(),
                "quality_path": publication.quality_path.as_posix(),
                "manifest_path": publication.manifest_path.as_posix(),
                "pointer_path": publication.pointer_path.as_posix(),
                "reused_existing": publication.reused_existing,
                "elapsed_minutes": round(
                    (time.monotonic() - started) / 60.0, 3
                ),
                "authorizations": publication.pointer["authorizations"],
            },
            sort_keys=True,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
