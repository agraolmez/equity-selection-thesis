#!/usr/bin/env python3
"""Independently verify the published development signal-date universe."""

from __future__ import annotations

import argparse
import json

from portfolio_bot.data.verify_phase_02_signal_date_universe import verify


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-publish", action="store_true")
    args = parser.parse_args()
    result = verify(publish=not args.no_publish)
    print(
        json.dumps(
            {
                "status": result["status"],
                "verification_id": result["verification_id"],
                "check_count": result["check_count"],
                "fail_count": result["fail_count"],
                "summary": result["report"]["summary"],
                "authorizations": result["report"]["authorizations"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result["status"] == "verified_passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
