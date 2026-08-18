#!/usr/bin/env python3
"""Independently verify and publish the Phase 2 CRSP return gate."""

from __future__ import annotations

import argparse
import json

from portfolio_bot.data.verify_phase_02_crsp_return_reconciliation import (
    verify_and_publish,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-publish", action="store_true")
    args = parser.parse_args()
    result = verify_and_publish(publish=not args.no_publish)
    report = result["report"]
    print(
        json.dumps(
            {
                "status": result["status"],
                "verification_id": result["verification_id"],
                "check_count": result["check_count"],
                "fail_count": result["fail_count"],
                "summary": report["summary"],
                "authorizations": report["authorizations"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result["status"] == "verified_passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
