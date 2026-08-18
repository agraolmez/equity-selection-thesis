#!/usr/bin/env python3
"""Build and publish the development-only Stage 8 risk/cost estimates."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from portfolio_bot.risk_cost.contract import DEFAULT_POLICY, validate_contract
from portfolio_bot.risk_cost.inputs import resolve_stage8_inputs
from portfolio_bot.risk_cost.materialization import materialize_risk_cost
from portfolio_bot.risk_cost.publication import publish_primary_risk_cost


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize Stage 8 point-in-time risk and conservative cost "
            "inputs. This command cannot train models or run a portfolio."
        )
    )
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    policy_path = args.policy.expanduser().resolve()
    started = time.monotonic()
    _, rules = validate_contract(policy_path)
    print("Stage 8.1: authenticating the closed development inputs...", flush=True)
    inputs = resolve_stage8_inputs(policy_path)
    print("Stage 8.2: computing risk/cost estimates with one return scan...", flush=True)

    last_update = 0.0

    def progress(message: str) -> None:
        nonlocal last_update
        now = time.monotonic()
        if now - last_update >= 10.0 or message.startswith("complete"):
            print(
                f"Stage 8 materialization: {message}; "
                f"elapsed={(now - started) / 60.0:.1f}m",
                flush=True,
            )
            last_update = now

    try:
        result = materialize_risk_cost(inputs, rules, progress=progress)
    except TypeError as exc:
        if "progress" not in str(exc):
            raise
        result = materialize_risk_cost(inputs, rules)

    print("Stage 8.3: publishing immutable primary output sets...", flush=True)
    publication = publish_primary_risk_cost(
        inputs,
        result,
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
                "elapsed_minutes": round((time.monotonic() - started) / 60.0, 3),
                "authorizations": publication.pointer["authorizations"],
            },
            sort_keys=True,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
